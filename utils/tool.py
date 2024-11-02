def RowToTextualRepresentation(row):
    data = [f"Tool: {val.post_title}\nDescription: {val.post_content}\n" for val in row]
    return data