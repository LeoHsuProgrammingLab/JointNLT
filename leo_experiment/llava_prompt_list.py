def generate_prompt_list(desc_template):
    prompt_list = [
        f"""
            Describe the object in the image in one sentence. 
            Please format the output as a single line.
        """,
        f"""
            This is the original description of tracked target object in the first frame: {desc_template}.
            Please do the following:
            1. Describe this object in current frame more concisely, updated to fit the current situation in one sentence.
            2. Output the bounding box coordinates of this object in [x1, y1, x2, y2] format. If the target object doesn't exist in current frame, please output coordinate [0, 0, 0, 0].
            Please format the output as a single line, with the description and bounding box separated by '#': updated_description #bounding_box
        """, 
        f"""
            Describe the object "{desc_template}" in the image with the following attributes:
            1. Object Color
            2. Object Name
            3. Object Action
            4. Object Movement
            5. Object Location
            Please format the output as a single line: color, name, action, movement, location, and the final description
        """,
    ]

    return prompt_list