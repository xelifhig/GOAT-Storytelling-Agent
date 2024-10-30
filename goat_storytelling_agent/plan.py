"""Unifies all plot forms such as by-chapter and by-scene outlines in a single dict."""
import re
import json


class Plan:
    @staticmethod
    def split_by_act(original_plan):
        # removes only Act texts with newline prepended somewhere near
        acts = re.split('\n.{0,5}?Act ', original_plan)
        # remove random short garbage from re split
        acts = [text.strip() for text in acts[:]
                if (text and (len(text.split()) > 3))]
    
        # Keep only first act or handle case where Act 1 is already in text
        if len(acts) >= 1:
            act = acts[0]
            if not act.startswith('Act '):
                act = 'Act 1: ' + act
            return [act]
        else:
            print('Fail: split_by_act', original_plan)
            return []

    @staticmethod
    def parse_act(act):
        act = re.split(r'\n.{0,20}?Chapter .+:', act.strip())
        chapters = [text.strip() for text in act[1:]
                    if (text and (len(text.split()) > 3))]
        return {'act_descr': act[0].strip(), 'chapters': chapters}

    @staticmethod
    def parse_text_plan(text_plan):
        acts = Plan.split_by_act(text_plan)
        if not acts:
            return []
        plan = [Plan.parse_act(act) for act in acts if act]
        plan = [act for act in plan if act['chapters'] and len(act['chapters']) == 4]  # Validate 4 chapters
        return plan

    @staticmethod
    def normalize_text_plan(text_plan):
        plan = Plan.parse_text_plan(text_plan)
        text_plan = Plan.plan_2_str(plan)
        return text_plan

    @staticmethod
    def act_2_str(plan, act_num):
        text_plan = ''
        chs = []
        ch_num = 1
        if plan and len(plan) > 0:  # Modified to handle single act
            act = plan[0]  # Always use first act
            act_descr = act['act_descr'] + '\n'
            if not re.search(r'Act \d', act_descr[0:50]):
                act_descr = 'Act 1:\n' + act_descr
            for chapter in act['chapters'][:4]:  # Limit to 4 chapters
                if act_num == 1:  # Modified since we only have Act 1
                    act_descr += f'- Chapter {ch_num}: {chapter}\n'
                    chs.append(ch_num)
                ch_num += 1
            text_plan += act_descr + '\n'
        return text_plan.strip(), chs

    @staticmethod
    def plan_2_str(plan):
        text_plan = ''
        ch_num = 1
        if plan and len(plan) > 0:  # Modified to handle single act
            act = plan[0]  # Always use first act
            act_descr = act['act_descr'] + '\n'
            if not re.search(r'Act \d', act_descr[0:50]):
                act_descr = 'Act 1:\n' + act_descr
            for chapter in act['chapters'][:4]:  # Limit to 4 chapters
                act_descr += f'- Chapter {ch_num}: {chapter}\n'
                ch_num += 1
            text_plan += act_descr + '\n'
        return text_plan.strip()

    @staticmethod
    def save_plan(plan, fpath):
        with open(fpath, 'w') as fp:
            json.dump(plan, fp, indent=4)
