system = (
    "You are a helpful assistant for fiction writing. "
    "Always cut the bullshit and provide concise outlines with useful details. "
    "Do not turn your stories into fairy tales, be realistic.")

book_spec_fields = ['Genre', 'Place', 'Time', 'Theme',
                    'Tone', 'Point of View', 'Characters', 'Premise']

book_spec_format = (
    "Genre: genre\n"
    "Place: place\n"
    "Time: period\n"
    "Theme: main topics\n"
    "Tone: tone\n"
    "Point of View: POV\n"
    "Characters: use specific names already\n"
    "Premise: describe some concrete events already")

scene_spec_format = (
    "Chapter [number]:\nScene [number]:\nCharacters: character list\nPlace: place\nTime: absolute or relative time\nEvent: what happens\nConflict: scene micro-conflict\n"
    "Story value: story value affected by the scene\nStory value charge: the charge of story value by the end of the scene (positive or negative)\nMood: mood\nOutcome: the result.")

prev_scene_intro = "\n\nHere is the ending of the previous scene:\n"
cur_scene_intro = "\n\nHere is the last written snippet of the current scene:\n"


def init_book_spec_messages(topic, form):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"Given the topic, come up with a specification to write a {form}. Write spec using the format below. "
                    f"Topic: {topic}\nFormat:\n\"\"\"\n{book_spec_format}\"\"\""},
    ]
    return messages


def missing_book_spec_messages(field, text_spec):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": (
            f"Given a hypothetical book spec, fill the missing field: {field}."
            f'Return only field, separator and value in one line like "Field: value".\n'
            f'Book spec:\n"""{text_spec}"""')
        }
    ]
    return messages


def enhance_book_spec_messages(book_spec, form):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"Make the specification for an upcoming {form} more detailed "
            f"(specific settings, major events that differentiate the {form} "
            f"from others). Do not change the format or add more fields."
            f"\nEarly {form} specification:\n\"\"\"{book_spec}\"\"\""}
    ]
    return messages


def create_plot_chapters_messages(book_spec, form):
    messages = [
        {"role": "user", "content": (
            f"Come up with a plot for a bestseller-grade {form} in exactly 1 act with exactly 4 chapters based on this specification:\n\"\"\"{book_spec}\"\"\"\n\n"
            "Break down the plot into chapters using the following structure:\nAct 1\n- Exactly Four Chapters\n"
            "Ensure all chapters are directly connected to the main premise and theme.")}
    ]
    return messages


def enhance_plot_chapters_messages(act_num, text_plan, book_spec, form):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Based on this book specification:\n\"\"\"{book_spec}\"\"\"\n\n"
            f"Come up with a plot for a bestseller-grade {form} in exactly 1 act with exactly 4 chapters. "
            "Break down the plot into chapters using the following structure:\nAct 1\n- Exactly Four Chapters\n"
            "Ensure each chapter directly relates to the main premise and theme.")},
        {"role": "assistant", "content": text_plan},
        {"role": "user", "content": (
            "Rewrite the plan for Act 1 so that chapter's story value alternates (i.e. if Chapter 1 is positive, Chapter 2 is negative, and so on). "
            "Describe only concrete events and actions (who did what). Make it very short (one brief sentence and value charge indication per chapter). "
            "Ensure there are exactly 4 chapters and each chapter clearly connects to the book's premise.")}
    ]
    return messages


def split_chapters_into_scenes_messages(act_num, text_act, form, book_spec):  # Added book_spec parameter
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Based on this book specification:\n\"\"\"{book_spec}\"\"\"\n\n"
            "Break each of the 4 chapters in Act 1 into scenes (number depends on how packed a chapter is), give scene specifications for each.\n"
            f"Here is the by-chapter plot summary for the act in a {form}:\n\"\"\"{text_act}\"\"\"\n\n"
            f"Scene spec format:\n\"\"\"{scene_spec_format}\"\"\"\n"
            "Ensure each scene directly relates to the main premise and theme.")}
    ]
    return messages


def scene_messages(scene, sc_num, ch_num, text_plan, form, book_spec):  # Added book_spec parameter
    messages = [
        {"role": "system", "content": 'You are an expert fiction writer. Write detailed scenes with lively dialogue that stay true to the original premise and theme.'},
        {"role": "user",
            "content": (
                f"Write a long detailed scene for a {form} for scene {sc_num} in chapter {ch_num} based on this specification:\n"
                f"\"\"\"{book_spec}\"\"\"\n\n"
                "Scene details:\n"
                f"\"\"\"{scene}\"\"\"\n\n"
                f"Overall plot:\n\"\"\"{text_plan}\"\"\"\n\n"
                "Be creative but ensure the scene directly connects to the main premise and theme. "
                "Maintain consistency with previous chapters and the overall story arc. "
                "Do NOT use foreshadowing.")},
        {"role": "assistant", "content": f"\nChapter {ch_num}, Scene {sc_num}\n"},
    ]
    return messages

def scene_summary_messages(scene_content, sc_num, ch_num):
    messages = [
        {"role": "system", "content": "Create concise, factual summaries of story scenes focusing on key events, character developments, and plot points."},
        {"role": "user", "content": (
            f"Create a brief summary (max 100 words) of the following scene from Chapter {ch_num}, Scene {sc_num}. "
            "Focus on key events, character developments, and details important for story continuity:\n"
            f"\"\"\"{scene_content}\"\"\""
        )}
    ]
    return messages

def scene_messages_with_history(scene, sc_num, ch_num, text_plan, form, book_spec, previous_summaries):
    summary_text = "\n".join([
        f"Chapter {summary['chapter']}, Scene {summary['scene']}: {summary['content']}"
        for summary in previous_summaries
    ])
    
    messages = [
        {"role": "system", "content": 'You are an expert fiction writer. Write detailed scenes with lively dialogue that stay true to the original premise and theme.'},
        {"role": "user",
            "content": (
                f"Write a long detailed scene for a {form} for scene {sc_num} in chapter {ch_num}.\n\n"
                f"Book specification:\n\"\"\"{book_spec}\"\"\"\n\n"
                "Scene details:\n"
                f"\"\"\"{scene}\"\"\"\n\n"
                f"Overall plot:\n\"\"\"{text_plan}\"\"\"\n\n"
                f"Previous scenes summary:\n\"\"\"{summary_text}\"\"\"\n\n"
                "Write a scene that maintains perfect continuity with previous events. "
                "Be creative but ensure the scene directly connects to the main premise and theme. "
                "Maintain consistency with previous chapters and the overall story arc. "
                "Do NOT use foreshadowing.")},
        {"role": "assistant", "content": f"\nChapter {ch_num}, Scene {sc_num}\n"},
    ]
    return messages