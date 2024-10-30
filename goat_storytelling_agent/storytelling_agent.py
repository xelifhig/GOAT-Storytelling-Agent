import sys
import time
import re
import json
import requests
import traceback
import utils
from plan import Plan


SUPPORTED_BACKENDS = ["hf", "llama.cpp", "openai"]


def generate_prompt_parts(
        messages, include_roles=set(('user', 'assistant', 'system'))):
    last_role = None
    messages = [m for m in messages if m['role'] in include_roles]
    for idx, message in enumerate(messages):
        nl = "\n" if idx > 0 else ""
        if message['role'] == 'system':
            if idx > 0 and last_role not in (None, "system"):
                raise ValueError("system message not at start")
            yield f"{message['content']}"
        elif message['role'] == 'user':
            yield f"{nl}### USER: {message['content']}"
        elif message['role'] == 'assistant':
            yield f"{nl}### ASSISTANT: {message['content']}"
        last_role = message['role']
    if last_role != 'assistant':
        yield '\n### ASSISTANT:'


def _query_chat_hf(endpoint, messages, tokenizer, retries=3,
                   request_timeout=120, max_tokens=4096,
                   extra_options={'do_sample': True}):
    endpoint = endpoint.rstrip('/')
    prompt = ''.join(generate_prompt_parts(messages))
    tokens = tokenizer(prompt, add_special_tokens=True,
                       truncation=False)['input_ids']
    data = {
        "inputs": prompt,
        "parameters": {
            'max_new_tokens': max_tokens - len(tokens),
            **extra_options
        }
    }
    headers = {'Content-Type': 'application/json'}

    while retries > 0:
        try:
            response = requests.post(
                f"{endpoint}/generate", headers=headers, data=json.dumps(data),
                timeout=request_timeout)
            if messages and messages[-1]["role"] == "assistant":
                result_prefix = messages[-1]["content"]
            else:
                result_prefix = ''
            generated_text = result_prefix + json.loads(
                response.text)['generated_text']
            return generated_text
        except Exception:
            traceback.print_exc()
            print('Timeout error, retrying...')
            retries -= 1
            time.sleep(5)
    else:
        return ''


def _query_chat_openai(endpoint, messages, retries=3,
                      request_timeout=600, max_tokens=4096,
                      extra_options={'model': 'gpt-3.5-turbo', 'api_key': '123'}):
    endpoint = endpoint.rstrip('/')
    if 'api_key' not in extra_options or not extra_options['api_key']:
        raise ValueError("OpenAI API key must be provided in extra_options")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {extra_options['api_key']}"
    }
    
    # Use the messages directly instead of generate_prompt_parts
    # OpenAI already expects the role-based format
    api_messages = []
    for message in messages:
        api_messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    data = {
        "model": extra_options.get('model', 'gpt-3.5-turbo'),
        "messages": api_messages,
        "max_tokens": max_tokens
    }
    
    # Add any additional OpenAI parameters from extra_options
    for key, value in extra_options.items():
        if key not in ['model', 'api_key']:
            data[key] = value

    while retries > 0:
        try:
            print(f"\n\n========== Submitting request to OpenAI: >>\n{json.dumps(data, indent=2)}")
            sys.stdout.flush()
            
            response = requests.post(
                f"{endpoint}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=request_timeout
            )
            
            response_json = response.json()
            print(f"\n========== Received response from OpenAI: >>\n{json.dumps(response_json, indent=2)}")
            sys.stdout.flush()
            
            # Check for API errors
            if 'error' in response_json:
                raise Exception(f"OpenAI API error: {response_json['error']}")
            
            # Check if response has expected structure
            if not response_json.get('choices') or len(response_json['choices']) == 0:
                raise Exception("No choices in OpenAI response")
                
            generated_text = response_json['choices'][0]['message']['content']
            
            if messages and messages[-1]["role"] == "assistant":
                result_prefix = messages[-1]["content"]
                generated_text = result_prefix + generated_text
                
            return generated_text
            
        except Exception:
            traceback.print_exc()
            print('Error occurred, retrying...')
            retries -= 1
            time.sleep(5)
    
    return ''
    

def _query_chat_llamacpp(endpoint, messages, retries=3, request_timeout=120,
                         max_tokens=4096, extra_options={}):
    endpoint = endpoint.rstrip('/')
    headers = {'Content-Type': 'application/json'}
    prompt = ''.join(generate_prompt_parts(messages))
    print(f"\n\n========== Submitting prompt: >>\n{prompt}", end="")
    sys.stdout.flush()
    response = requests.post(
        f"{endpoint}/tokenize", headers=headers,
        data=json.dumps({"content": prompt}),
        timeout=request_timeout, stream=False)
    tokens = [1, *response.json()["tokens"]]
    data = {
        "prompt": tokens,
        "stream": True,
        "n_predict": max_tokens - len(tokens),
        **extra_options,
    }
    jdata = json.dumps(data)
    request_kwargs = dict(headers=headers, data=jdata,
                          timeout=request_timeout, stream=True)
    response = requests.post(f"{endpoint}/completion", **request_kwargs)
    result = bytearray()
    if messages and messages[-1]["role"] == "assistant":
        result += messages[-1]["content"].encode("utf-8")
    is_first = True
    for line in response.iter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"error:"):
            retries -= 1
            print(f"\nError(retry={retries}): {line!r}")
            if retries < 0:
                break
            del response
            time.sleep(5)
            response = requests.post(f"{endpoint}/completion", **request_kwargs)
            is_first = True
            result.clear()
            continue
        if not line.startswith(b"data: "):
            raise ValueError(f"Got unexpected response: {line!r}")
        parsed = json.loads(line[6:])
        content = parsed.get("content", b"")
        result += bytes(content, encoding="utf-8")
        if is_first:
            is_first = False
            print("<<|", end="")
            sys.stdout.flush()
        print(content, end="")
        sys.stdout.flush()
        if parsed.get("stop") is True:
            break
    print("\nDone reading response.")
    return str(result, encoding="utf-8").strip()


class StoryAgent:
    def __init__(self, backend_uri, backend="hf", request_timeout=120,
                 max_tokens=4096, n_crop_previous=400,
                 prompt_engine=None, form='novel',
                 extra_options={}, scene_extra_options={}):

        self.backend = backend.lower()
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError("Unknown backend")

        if self.backend == "hf":
            from transformers import LlamaTokenizerFast
            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                "GOAT-AI/GOAT-70B-Storytelling")

        if prompt_engine is None:
            import prompts
            self.prompt_engine = prompts
        else:
            self.prompt_engine = prompt_engine

        self.form = form
        self.max_tokens = max_tokens
        self.extra_options = extra_options
        self.scene_extra_options = extra_options.copy()
        self.scene_extra_options.update(scene_extra_options)
        self.backend_uri = backend_uri
        self.n_crop_previous = n_crop_previous
        self.request_timeout = request_timeout

    def query_chat(self, messages, retries=3):
        if self.backend == "hf":
            result = _query_chat_hf(
                self.backend_uri, messages, self.tokenizer, retries=retries,
                request_timeout=self.request_timeout,
                max_tokens=self.max_tokens, extra_options=self.extra_options)
        elif self.backend == "llama.cpp":
            result = _query_chat_llamacpp(
                self.backend_uri, messages, retries=retries,
                request_timeout=self.request_timeout,
                max_tokens=self.max_tokens, extra_options=self.extra_options)
        elif self.backend == "openai":
            result = _query_chat_openai(
                self.backend_uri, messages, retries=retries,
                request_timeout=self.request_timeout,
                max_tokens=self.max_tokens)
        return result

    def parse_book_spec(self, text_spec):
        # Initialize book spec dict with empty fields
        fields = self.prompt_engine.book_spec_fields
        spec_dict = {field: '' for field in fields}
        last_field = None
        if "\"\"\"" in text_spec[:int(len(text_spec)/2)]:
            header, sep, text_spec = text_spec.partition("\"\"\"")
        text_spec = text_spec.strip()

        # Process raw spec into dict
        for line in text_spec.split('\n'):
            pseudokey, sep, value = line.partition(':')
            pseudokey = pseudokey.lower().strip()
            matched_key = [key for key in fields
                           if (key.lower().strip() in pseudokey)
                           and (len(pseudokey) < (2 * len(key.strip())))]
            if (':' in line) and (len(matched_key) == 1):
                last_field = matched_key[0]
                if last_field in spec_dict:
                    spec_dict[last_field] += value.strip()
            elif ':' in line:
                last_field = 'other'
                spec_dict[last_field] = ''
            else:
                if last_field:
                    # If line does not contain ':' it should be
                    # the continuation of the last field's value
                    spec_dict[last_field] += ' ' + line.strip()
        spec_dict.pop('other', None)
        return spec_dict

    def init_book_spec(self, topic):
        """Creates initial book specification

        Parameters
        ----------
        topic : str
            Short initial topic

        Returns
        -------
        List[Dict]
            Used messages for logging
        str
            Book specification text
        """
        messages = self.prompt_engine.init_book_spec_messages(topic, self.form)
        text_spec = self.query_chat(messages)
        spec_dict = self.parse_book_spec(text_spec)

        text_spec = "\n".join(f"{key}: {value}"
                              for key, value in spec_dict.items())
        # Check and fill in missing fields
        for field in self.prompt_engine.book_spec_fields:
            while not spec_dict[field]:
                messages = self.prompt_engine.missing_book_spec_messages(
                    field, text_spec)
                missing_part = self.query_chat(messages)
                key, sep, value = missing_part.partition(':')
                if key.lower().strip() == field.lower().strip():
                    spec_dict[field] = value.strip()
        text_spec = "\n".join(f"{key}: {value}"
                              for key, value in spec_dict.items())
        return messages, text_spec

    def enhance_book_spec(self, book_spec):
        """Make book specification more detailed

        Parameters
        ----------
        book_spec : str
            Book specification

        Returns
        -------
        List[Dict]
            Used messages for logging
        str
            Book specification text
        """
        messages = self.prompt_engine.enhance_book_spec_messages(
            book_spec, self.form)
        text_spec = self.query_chat(messages)
        spec_dict_old = self.parse_book_spec(book_spec)
        spec_dict_new = self.parse_book_spec(text_spec)

        # Check and fill in missing fields
        for field in self.prompt_engine.book_spec_fields:
            if not spec_dict_new[field]:
                spec_dict_new[field] = spec_dict_old[field]

        text_spec = "\n".join(f"{key}: {value}"
                              for key, value in spec_dict_new.items())
        return messages, text_spec

    def create_plot_chapters(self, book_spec):
        """Create initial by-plot outline of form

        Parameters
        ----------
        book_spec : str
            Book specification

        Returns
        -------
        List[Dict]
            Used messages for logging
        dict
            Dict with book plan
        """
        messages = self.prompt_engine.create_plot_chapters_messages(book_spec, self.form)
        plan = []
        while not plan:
            text_plan = self.query_chat(messages)
            if text_plan:
                plan = Plan.parse_text_plan(text_plan)
                # Validate one act with exactly 4 chapters
                if len(plan) == 1 and len(plan[0]['chapters']) == 4:
                    break
                plan = []
        return messages, plan

    def enhance_plot_chapters(self, book_spec, plan):
        """Enhances the outline to make the flow more engaging

        Parameters
        ----------
        book_spec : str
            Book specification
        plan : Dict
            Dict with book plannn

        Returns
        -------
        List[Dict]
            Used messages for logging
        dict
            Dict with updated book plan
        """
        text_plan = Plan.plan_2_str(plan)
        messages = self.prompt_engine.enhance_plot_chapters_messages(
            0, text_plan, book_spec, self.form)  # Always act 0 since we only have one act
        act = self.query_chat(messages)
        all_messages = [messages]
    
        if act:
            act_dict = Plan.parse_act(act)
            # Keep querying until we get exactly 4 chapters
            while len(act_dict['chapters']) != 4:
                act = self.query_chat(messages)
                act_dict = Plan.parse_act(act)
            plan[0] = act_dict
        
        return all_messages, plan

    def split_chapters_into_scenes(self, plan, book_spec):  # Added book_spec parameter
        all_messages = []
        act_chapters = {}
    
        # Modified to handle only one act
        i = 1  # Act 1
        act = plan[0]
        text_act, chs = Plan.act_2_str(plan, i)
        act_chapters[i] = chs[:4]  # Limit to 4 chapters
        messages = self.prompt_engine.split_chapters_into_scenes_messages(
            i, text_act, self.form, book_spec)  # Added book_spec
        act_scenes = self.query_chat(messages)
        act['act_scenes'] = act_scenes
        all_messages.append(messages)

        # Process scenes for the single act
        act_scenes = act['act_scenes']
        act_scenes = re.split(r'Chapter (\d+)', act_scenes.strip())

        act['chapter_scenes'] = {}
        chapters = [text.strip() for text in act_scenes[:]
                    if (text and text.strip())]
        current_ch = None
        merged_chapters = {}
        for snippet in chapters:
            if snippet.isnumeric():
                ch_num = int(snippet)
                if ch_num != current_ch and ch_num <= 4:  # Only process first 4 chapters
                    current_ch = snippet
                    merged_chapters[ch_num] = ''
                continue
            if merged_chapters and current_ch and int(current_ch) <= 4:
                merged_chapters[ch_num] += snippet
        ch_nums = [i for i in range(1, 5)]  # Always use chapters 1-4
        merged_chapters = {ch_num: merged_chapters.get(ch_num, '')
                            for ch_num in ch_nums}
        for ch_num, chapter in merged_chapters.items():
            scenes = re.split(r'Scene \d+.{0,10}?:', chapter)
            scenes = [text.strip() for text in scenes[1:]
                        if (text and (len(text.split()) > 3))]
            if not scenes:
                continue
            act['chapter_scenes'][ch_num] = scenes
    
        return all_messages, plan

    @staticmethod
    def prepare_scene_text(text):
        if not text:
            return ""
        
        lines = text.split('\n')
        if not lines:
            return ""
        
        # Find last "Chapter" marker in the beginning
        ch_ids = [i for i, line in enumerate(lines[:min(5, len(lines))])
                  if 'Chapter ' in line]
        if ch_ids:
            lines = lines[ch_ids[-1]+1:]
        
        # Find last "Scene" marker in the beginning
        if lines:  # Check if we still have lines after previous operation
            sc_ids = [i for i, line in enumerate(lines[:min(5, len(lines))])
                      if 'Scene ' in line]
            if sc_ids:
                lines = lines[sc_ids[-1]+1:]
            
        # Find first occurrence of Chapter/Scene marker in the rest of the text
        placeholder_i = None
        if lines:  # Check if we still have lines
            for i, line in enumerate(lines):
                if line.startswith('Chapter ') or line.startswith('Scene '):
                    placeholder_i = i
                    break
                
            if placeholder_i is not None:
                lines = lines[:placeholder_i]
            
        text = '\n'.join(lines)
        return text.strip()
    

    def write_a_scene(
            self, scene, sc_num, ch_num, plan, book_spec,  # Added book_spec parameter
            previous_scene=None):
        text_plan = Plan.plan_2_str(plan)
        messages = self.prompt_engine.scene_messages(
            scene, sc_num, ch_num, text_plan, self.form, book_spec)  # Added book_spec
        if previous_scene:
            previous_scene = utils.keep_last_n_words(previous_scene,
                                                        n=self.n_crop_previous)
            messages[1]['content'] += f'{self.prompt_engine.prev_scene_intro}\"\"\"{previous_scene}\"\"\"'
        generated_scene = self.query_chat(messages)
        generated_scene = self.prepare_scene_text(generated_scene)
        return messages, generated_scene


    def continue_a_scene(self, scene, sc_num, ch_num,
                         plan, book_spec,  # Added book_spec parameter
                         current_scene=None):
        text_plan = Plan.plan_2_str(plan)
        messages = self.prompt_engine.scene_messages(
            scene, sc_num, ch_num, text_plan, self.form, book_spec)  # Added book_spec
        if current_scene:
            current_scene = utils.keep_last_n_words(current_scene,
                                                    n=self.n_crop_previous)
            messages[1]['content'] += f'{self.prompt_engine.cur_scene_intro}\"\"\"{current_scene}\"\"\"'
        generated_scene = self.query_chat(messages)
        generated_scene = self.prepare_scene_text(generated_scene)
        return messages, generated_scene


    def generate_story(self, topic):
        """Example pipeline for a novel creation"""
        _, book_spec = self.init_book_spec(topic)
        _, book_spec = self.enhance_book_spec(book_spec)
        _, plan = self.create_plot_chapters(book_spec)
        _, plan = self.enhance_plot_chapters(book_spec, plan)
        _, plan = self.split_chapters_into_scenes(plan, book_spec)  # Added book_spec

        form_text = []
        # Modified to handle only one act with 4 chapters
        act = plan[0]
        for ch_num in range(1, 5):  # Always process chapters 1-4
            if ch_num in act['chapter_scenes']:
                chapter = act['chapter_scenes'][ch_num]
                sc_num = 1
                for scene in chapter:
                    previous_scene = form_text[-1] if form_text else None
                    _, generated_scene = self.write_a_scene(
                        scene, sc_num, ch_num, plan,
                        book_spec,  # Added book_spec
                        previous_scene=previous_scene)
                    form_text.append(generated_scene)
                    sc_num += 1
        return form_text
