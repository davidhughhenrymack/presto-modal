import os
import time
import yaml
from pathlib import Path

import modal
from fastapi.responses import StreamingResponse

from .common import app, vllm_image, Colors, MINUTES, VOLUME_CONFIG

INFERENCE_GPU_CONFIG = os.environ.get("INFERENCE_GPU_CONFIG", "a10g:2")
if len(INFERENCE_GPU_CONFIG.split(":")) <= 1:
    N_INFERENCE_GPUS = int(os.environ.get("N_INFERENCE_GPUS", 2))
    INFERENCE_GPU_CONFIG = f"{INFERENCE_GPU_CONFIG}:{N_INFERENCE_GPUS}"
else:
    N_INFERENCE_GPUS = int(INFERENCE_GPU_CONFIG.split(":")[-1])


with vllm_image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid


def get_model_path_from_run(path: Path) -> Path:
    with (path / "config.yml").open() as f:
        return path / yaml.safe_load(f.read())["output_dir"] / "merged"


@app.cls(
    gpu=INFERENCE_GPU_CONFIG,
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=15 * MINUTES,
)
class Inference:
    def __init__(self, run_name: str = "", run_dir: str = "/runs") -> None:
        self.run_name = run_name
        self.run_dir = run_dir

    @modal.enter()
    def init(self):
        if self.run_name:
            path = Path(self.run_dir) / self.run_name
            VOLUME_CONFIG[self.run_dir].reload()
            model_path = get_model_path_from_run(path)
        else:
            # Pick the last run automatically
            run_paths = list(Path(self.run_dir).iterdir())
            for path in sorted(run_paths, reverse=True):
                model_path = get_model_path_from_run(path)
                if model_path.exists():
                    break

        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Initializing vLLM engine for model at {model_path}",
            Colors.END,
            sep="",
        )

        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=N_INFERENCE_GPUS,
            disable_custom_all_reduce=True,  # brittle as of v0.5.0
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _stream(self, input: str):
        if not input:
            return

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=4 * 1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                request_output.outputs[0].text
                and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(
            Colors.GREEN,
            Colors.BOLD,
            f"🧠: Effective throughput of {throughput:.2f} tok/s",
            Colors.END,
            sep="",
        )

    @modal.method()
    async def completion(self, input: str):
        async for text in self._stream(input):
            yield text

    @modal.method()
    async def non_streaming(self, input: str):
        output = [text async for text in self._stream(input)]
        return "".join(output)

    @modal.web_endpoint()
    async def web(self, input: str):
        return StreamingResponse(self._stream(input), media_type="text/event-stream")

    @modal.exit()
    def stop_engine(self):
        if N_INFERENCE_GPUS > 1:
            import ray

            ray.shutdown()

        # access private attribute to ensure graceful termination
        self.engine._background_loop_unshielded.cancel()


@app.local_entrypoint()
def inference_main(run_name: str = "", prompt: str = ""):

    prompt = """[INST]Please replace the text to say \'The ripe adapter fears pilaf The spicy cast amuses canopy The rightful\' in this svg, return just the svg and no explanation. \n\n<?xml version=\'1.0\' encoding=\'utf8\'?>\n<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080" fill="none">\n\t<g id="slide_title_long">\n\t\t<rect width="1920" height="1080" fill="#F40000" id="!!!bg" />\n\t\t<image href="https://wcyb5qfzc4ygdwpj.public.blob.vercel-storage.com/image-extract/0730e0380cbaa4582444fdfb17bb1e9f-oYFr3viQMWRJaTD6IL68SJ7XCLunys.png" id="logo-coca-cola-image" x="214" y="224" width="413" height="129" alt="This image appears to be entirely blank or white. It\'s a solid white rectangular area without any visible content, patterns, or distinguishing features. The image might not have loaded correctly, or it could be intentionally blank." preserveaspectratio="xMidYMid meet" />\n\t\t<text id="7c848dad-657a-4b11-96bb-aaaff867fdd4" fill="white" xml:space="preserve" style="white-space: pre" font-family="TCCC-UnityHeadline" font-size="142" font-weight="bold" letter-spacing="0em">\n\t\t\t<tspan x="214" y="559.64" id="c1d03ac2-ecd0-4a9a-9120-3e046f4bebdd">Slide title that </tspan>\n\t\t\t<tspan x="214" y="693.64" id="8fa98230-f1ef-4ca4-9e60-815f3684dcfa">wraps on two lines</tspan>\n\t\t</text>\n\t\t<text id="edba6974-59eb-459d-ba09-d6c91110d49a" fill="white" xml:space="preserve" style="white-space: pre" font-family="TCCC-UnityText" font-size="65" letter-spacing="0em">\n\t\t\t<tspan x="214" y="802.3" id="f8ffff69-0cb8-4666-886c-4c8d94a84302">Subtitle text</tspan>\n\t\t</text>\n\t</g>\n\t<defs id="00f8g22t">\n\n\n</defs>\n</svg>[/INST]"""

    if not prompt:
        prompt = input(
            "Enter a prompt (including the prompt template, e.g. [INST] ... [/INST]):\n"
        )
    print(
        Colors.GREEN, Colors.BOLD, f"🧠: Querying model {run_name}", Colors.END, sep=""
    )
    response = ""
    print(Colors.GREEN, f"👤", sep="", end="")
    for chunk in Inference(run_name).completion.remote_gen(prompt):
        # response += chunk  # not streaming to avoid mixing with server logs
        print(chunk, end="")

    print(Colors.END)
    # print(Colors.BLUE, f"👤: {prompt}", Colors.END, sep="")
    # print(Colors.GRAY, f"🤖: {response}", Colors.END, sep="")
