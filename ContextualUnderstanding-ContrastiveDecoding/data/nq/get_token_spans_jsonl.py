import jsonlines

INPUT_FILE = "v1.0-simplified_nq-dev-all.jsonl"
START_TOKEN = 3521
END_TOKEN = 3525
QAS_ID = 4549465242785278785
REMOVE_HTML = True


def get_span_from_token_offsets(f, start_token, end_token, qas_id,
                                remove_html):
    for obj in f:
        if obj["example_id"] != qas_id:
            continue

        if remove_html:
            answer_span = [
                item["token"]
                for item in obj["document_tokens"][start_token:end_token]
                if not item["html_token"]
            ]
        else:
            answer_span = [
                item["token"]
                for item in obj["document_tokens"][start_token:end_token]
            ]

        return " ".join(answer_span)


with jsonlines.open(INPUT_FILE) as f:
    result = get_span_from_token_offsets(f, START_TOKEN, END_TOKEN, QAS_ID,
                                         REMOVE_HTML)

print(result)