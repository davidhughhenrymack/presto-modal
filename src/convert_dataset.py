def conv_openai_msg(msg):
    return {
        "from": "human" if msg["role"] == "user" else "assistant",
        "value": msg["content"],
    }


def main():

    import json

    # Read the JSONL file
    with open("data/apply_template_train_200_openai.jsonl", "r") as file:
        data = [json.loads(line) for line in file]

    # Print the number of records read
    print(f"Read {len(data)} records from the JSONL file.")

    # Initialize a list to store the reformatted data
    reformatted_data = []

    # Reformat each record to ShareGPT format
    for record in data:
        reformatted_record = {
            "conversations": [conv_openai_msg(i) for i in record["messages"]]
        }
        reformatted_data.append(reformatted_record)

    # Write the reformatted data to a new JSONL file
    output_file = "data/apply_template_train_200_sharegpt.jsonl"
    with open(output_file, "w") as file:
        for record in reformatted_data:
            json.dump(record, file)
            file.write("\n")

    print(f"Reformatted {len(reformatted_data)} records and saved to {output_file}.")


if __name__ == "__main__":
    main()
