import os
import re
import logging
import json
import argparse
import requests
import io
import base64
from PIL import Image, PngImagePlugin
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class PromptGenerator():
    def __init__(self, ori_prompt):
        self.ori_prompt = ori_prompt
        default_args = {
            'prompt': {
                'type': str,
                'default': self.ori_prompt
            },
            'temperature': {
                'type': float,
                'default': 0.3,
                'range': [0, 1]
            },
            'top_k': {
                'type': int,
                'default': 8,
                'range': [1, 200]
            },
            'max_length': {
                'type': int,
                'default': 80,
                'range': [1, 200]
            },
            'repetition_penalty': {
                'type': float,
                'default': 1.2,
                'range': [0, 10]
            },
            'num_return_sequences': {
                'type': int,
                'default': 1,
                'range': [1, 5]
            },
        }

        self.parser = argparse.ArgumentParser()

        # optional arguments
        for arg, def_arg in default_args.items():
            arg = '--' + arg
            self.parser.add_argument(arg, type=def_arg['type'], default=def_arg['default'], required=False)

    # def get_blacklist(self):
    #     """Check and load blacklist
    #
    #     Returns:
    #         list: List of terms from the blacklist dictionary
    #     """
    #     blacklist_filename = 'blacklist.txt'
    #     blacklist = []
    #     if not os.path.exists(blacklist_filename):
    #         logging.warning("Blacklist file missing: %s", blacklist_filename)
    #         return blacklist
    #     with open(blacklist_filename, 'r') as f:
    #         for line in f:
    #             blacklist.append(line)
    #
    #         return blacklist

    def simple(self):
        import random
        path = os.path.join(os.getcwd(), 'prompt3.txt')
        prompt = open(path, encoding='utf-8').read().splitlines()
        generated = []
        num_word = 6
        non_artists = [art for art in prompt if not art.startswith("art by")]
        # artists = [art for art in prompt if art.startswith("art by")]
        while len(sorted(set(generated), key=lambda d: generated.index(d))) < num_word:
            rand = random.randint(0, len(non_artists)-1)
            generated.append(non_artists[rand])
        # generated.append(artists[random.randint(0,len(artists))])
        generated = ', '.join(sorted(set(generated), key=lambda d: generated.index(d)))
        generated = self.ori_prompt + ', ' + generated
        return generated

    def fixed(self):
        string = "assasins creed art style, smoke, dark fantasy, art by Glennray Tutor"
        generated = self.ori_prompt + ', ' + string
        return generated

    def post(self):
        """Post method

        Returns:
            string: JSON list with the generated prompts
        """
        args = self.parser.parse_args()
        # self.validate_args(args)

        prompt = args.prompt
        temperature = args.temperature
        top_k = args.top_k
        max_length = args.max_length
        num_return_sequences = args.num_return_sequences
        repetition_penalty = args.repetition_penalty
        # request_uuid = uuid.uuid4()
        try:
            # build model
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion-v2')
        except Exception as e:
            logging.error(
                "Exception encountered while attempting to install tokenizer: %s", e)
            # abort(500, message="There was an error processing your request")
        try:
            # generate prompt
            logging.debug("Generate new prompt from: \"%s\"", prompt)
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
            output = model.generate(input_ids, do_sample=True, temperature=temperature,
                                    top_k=top_k, max_length=max_length,
                                    num_return_sequences=num_return_sequences,
                                    repetition_penalty=repetition_penalty,
                                    penalty_alpha=0.6, no_repeat_ngram_size=1,
                                    early_stopping=True)
            prompt_output = []
            # blacklist = self.get_blacklist()
            for count, value in enumerate(output):
                prompt_output.append(
                    tokenizer.decode(value, skip_special_tokens=True)
                )
                # for term in blacklist:
                #     prompt_output[count] = re.sub(
                #         term, "", prompt_output[count], flags=re.IGNORECASE)

            return prompt_output

        except Exception as e:
            logging.error(
                "Exception encountered while attempting to generate prompt: %s", e)
            # abort(500, message="There was an error processing your request")