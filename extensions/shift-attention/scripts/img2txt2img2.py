# Shift Attention script for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/yownas/shift-attention
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.
# Will also support multiple numbers. "(cat:1~0~1)" will go from cat:1 to cat:0 to cat:1 streched over the number of steps
import time, datetime
import gradio as gr
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import re
import sys
import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms
from torch.nn import functional as F
import modules.scripts as scripts
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state, sd_upscalers
from modules.images import resize_image
import modules.shared as shared

from modules.test_swinIR import run_swinIR
from modules.main_test_bsrgan import run_bsr
from modules.inference_realesrgan import run_esr
from modules.crawl import crawl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rife.RIFE_HDv3 import Model

__ = lambda key, value=None: opts.data.get(f'customscript/shift-attention.py/txt2img/{key}/value', value)

DEFAULT_UPSCALE_METH = __('Upscaler', 'R-ESRGAN 4x+')
DEFAULT_UPSCALE_RATIO = __('Upscale ratio', 1.0)
CHOICES_UPSCALER = [x.name for x in sd_upscalers]

a,b,c,d = crawl()
def curve_steps(curve, curvestr, steps):
    strengths = []
    for i in range(steps+1):
        strength = float(i / float(steps))

        # Calculate curve
        if curve == "Hug-the-middle":
            # https://www.wolframalpha.com/input?i=graph+x%2B%28s%2F30%29*sin%28x*pi*2%29+from+0+to+1%2C+s%3D3
            strength = strength + (curvestr / 30.0 * math.sin(strength * 2 * math.pi))
        elif curve == "Hug-the-nodes":
            # https://www.wolframalpha.com/input?i=graph+x-%28s%2F30%29*sin%28x*pi*2%29+from+0+to+1%2C+s%3D3
            strength = strength - (curvestr / 30.0 * math.sin(strength * 2 * math.pi))
        elif curve == "Slow start":
            # https://www.wolframalpha.com/input?i=graph+x%5Es+from+0+to+1%2C+s%3D3
            strength = strength ** curvestr
        elif curve == "Quick start":
            # https://www.wolframalpha.com/input?i=graph+%281-x%29%5Es+from+0+to+1%2C+s%3D3
            strength = -(1 - strength) ** curvestr + 1
        elif curve == "Easy ease in":
            # https://www.wolframalpha.com/input?i=graph+%281-cos%28x%5E%28s*pi%2F10%29*pi%29%29%2F2+from+0+to+1%2C+s%3D3
            strength = (1 - math.cos(strength ** (curvestr * math.pi) * math.pi)) / 2.0
        elif curve == "Partial":
            # "Travel" part way before switching to next seed
            strength = strength * curvestr / 10.0
        elif curve == "Random":
            # Random flicker before switching to next seed
            strength = random.uniform(0, curvestr / 10.0)
        elif curve == "Gaussian":
            mu = 0.5
            sigma = 0.3
            strength = math.exp(-((strength - mu) / sigma) ** 2) * curvestr

        elif curve == "Beta":
            alpha = 2
            beta = 2
            strength = strength ** (alpha - 1) * (1 - strength) ** (beta - 1) * curvestr

        elif curve == 'z':
            # curvestr = 10
            # https://www.wolframalpha.com/input?i=graph+%281%2F%281%2Bexp%28-1*s*%28x-0.5%29%29%29%29+from+0+to+1+where+s%3D8
            strength = 1/(1+np.exp(-1*curvestr*(strength-0.5)))

        elif curve == "Logit":
            # https://www.wolframalpha.com/input?i=graph+%28ln%28x%2F%281-x%29%29%2F10%2B0.5%29+from+0+to+1
            curvestr = 10
            if len(strengths) == 0:
                strength = 0
            elif len(strengths) == steps:
                strength = 1
            else:
                strength = np.log(strength/(1-strength+0.0001))/curvestr + 0.5

        strengths.append(strength)
    return strengths

class Script(scripts.Script):
    def title(self):
        return "Interpolation2"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Accordion(label='Varialbe Explanations...', open=True):
            gr.Markdown("""
                * Steps : 텍스트 와 텍스트 사이에 몇 개의 기본 interpolation step을 만들 것인지에 대한 것. <br> 예를 들어 steps가 10 이면 [0,0.1,0.2...0.9,1] 로 10개의 구간이 만들어집니다. 
                * SSIM minimum: Frame을 새로 만들 때 SSIM minimum 보다 더 적은 ssim 을 가진 frame 들은 그냥 drop 해버립니다. 
                * SSIM threshold: Frame이 모두 만들어진 후, SSIM threshold 보다 적은 ssim 을 가진 frame이 있으면 그 사이를 다시 반으로 쪼개서 새로운 프레임을 생성합니다.
                * RIFE passes: RIFE를 몇 번이나 적용할 것인가?
                * Curve function: 구간을 나눌 때 어떤 curve function 을 따라서 나눌 것인가? 예를 들어 Linear 이면 균일하게 나누고, Sigmoid 이면 시그모이드 함수 를 따르게 나눔. 
            """)

        steps = gr.Number(label='Steps', value=29)

        with gr.Row():
            video_fps = gr.Number(label='FPS', value=10.0)
            lead_inout = gr.Number(label='Lead in/out', value=0, visible=False)

        with gr.Row():
            ssim_min = gr.Slider(label='SSIM minimum', value=0.5, minimum=0.0, maximum=1.0, step=0.1)
            ssim_diff = gr.Slider(label='SSIM threshold', value=0.0, minimum=0.0, maximum=1.0, step=0.1)
            ssim_ccrop = gr.Slider(label='SSIM CenterCrop%', value=0, minimum=0, maximum=100, step=1)

        with gr.Row():
            rife_passes = gr.Number(label='RIFE passes', value=3)
            rife_drop = gr.Checkbox(label='Drop original frames', value=False, visible=False)

        with gr.Row():
            txt1 = gr.Textbox(value=list(a), lines=3)
            txt2 = gr.Textbox(value=list(b), lines=3)
            txt3 = gr.Textbox(value=list(c), lines=3)
            txt4 = gr.Textbox(value=list(d), lines=3)
            txt5 = gr.Textbox(value="", lines=3)
            txt6 = gr.Textbox(value="", lines=3)
            txt7 = gr.Textbox(value="", lines=3)
            txt8 = gr.Textbox(value="", lines=3)
            txt9 = gr.Textbox(value="", lines=3)
            txt10 = gr.Textbox(value="", lines=3)


        with gr.Row():
            curve = gr.Dropdown(label='Curve function', value='Linear', choices=[
                'Linear', 'Hug-the-middle', 'Hug-the-nodes', 'Slow start', 'Quick start', 'Easy ease in', 'Partial',
                'Random', 'Gaussian', 'Beta', 'Sigmoid', "Logit"
            ])
            curvestr = gr.Slider(label='Rate strength', value=3.0, minimum=0.0, maximum=10.0, step=0.1)



        with gr.Row():
            order_frames = gr.Checkbox(label='order_frames', value=True)
            interpolate_latent = gr.Checkbox(label='Interpolate in latent', value=True)
            add_prompt = gr.Checkbox(label='Add Prompt', value=True)

        with gr.Accordion(label='Super Resolution', open=True, visible=True):
            upscale_meth = gr.Dropdown(label='Upscaler', value=lambda: DEFAULT_UPSCALE_METH,
                                       choices=CHOICES_UPSCALER)
            upscale_ratio = gr.Slider(label='Upscale ratio', value=lambda: DEFAULT_UPSCALE_RATIO, minimum=0.0,
                                      maximum=8.0, step=0.1)

        with gr.Accordion(label='Shift Attention Extras...', open=False, visible=False):
            gr.HTML(value='Shift Attention links: <a href=http://github.com/yownas/shift-attention/>Github</a>')

            with gr.Row():
                show_images = gr.Checkbox(label='Show generated images in ui', value=True)
            substep_min = gr.Number(label='SSIM minimum step', value=0.0001)
            ssim_diff_min = gr.Slider(label='SSIM min threshold', value=75, minimum=0, maximum=100, step=1)
            save_stats = gr.Checkbox(label='Save extra status information', value=True)


        return [txt1, txt2, txt3, txt4, txt5, txt6, txt7, txt8, txt9, txt10, steps, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_min, ssim_diff, ssim_ccrop,
                ssim_diff_min, substep_min, rife_passes, rife_drop, save_stats, order_frames, interpolate_latent, add_prompt, curve, curvestr]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    # def get_text_from_image(self, images):
    #     prompts = []
    #     for idx,img in enumerate(images):
    #         if img:
    #             prompts.append(shared.interrogator.interrogate(img.convert("RGB")))
    #             if idx==0:
    #                 prompts[-1]+=":1~0 AND "
    #             elif idx==len(images)-1:
    #                 prompts[-1] = prompts[-1] + ":0~1"
    #             else:
    #                 prompts[-1] = prompts[-1] + ":0~0.5 THEN " + prompts[-1] + ":0.5~0 AND "
    #
    #     return prompts

    def get_prompt_from_texts(self, texts, add_prompt):
        # if add_prompt:
        #     texts = [self.prompt_generator(txt) for txt in texts if txt]
        # else:
        texts = [txt for txt in texts if txt]

        for idx, text in enumerate(texts):
            if text:
                if idx==0:
                    texts[idx]+=':1~0 AND '
                elif idx==len(texts)-1:
                    texts[idx] = texts[idx] + ":0~1"
                else:
                    texts[idx] = texts[idx] + ":0~1 THEN " + texts[idx] + ":1~0 AND "

        return ''.join(texts)

    def prompt_generator(self, prompt, num=1):
        with open('C:\\Users\\chsjk\\PycharmProjects\\blacklist.txt') as f:
            black_lists = [word for word in f.read().split('\n') if word]

        with open('C:\\Users\\chsjk\\PycharmProjects\\whitelist.txt') as f:
            white_list = ', '.join(f.read().split('\n'))
        file_path = os.path.join(os.getcwd(), "interrogate")
        # folders = os.listdir(file_path)
        folders = ['flavors_short.txt', 'mediums.txt', 'movements.txt']
        for folder in folders:
            text_f = os.path.join(file_path, folder)
            with open(text_f, 'r', encoding='utf-8') as f:
                words_list = [word for word in f.read().split('\n') if word]
                if folder == 'flavors_short.txt':
                    num_a = num + 1
                else:
                    num_a = num
                for _ in range(num_a):
                    word = random.choice(words_list)
                    while word=='' or word in black_lists:
                        word = random.choice(words_list)
                    prompt += ', ' + word
        prompt += white_list
        return prompt

    def run(self, p, txt1, txt2, txt3, txt4, txt5, txt6, txt7, txt8, txt9, txt10, steps, video_fps, show_images, lead_inout, upscale_meth, upscale_ratio, ssim_min, ssim_diff, ssim_ccrop,
            ssim_diff_min, substep_min, rife_passes, rife_drop, save_stats, order_frames, interpolate_latent, add_prompt, curve, curvestr):
        start = time.time()
        re_attention_span = re.compile(r"([\-.\d]+~[\-~.\d]+)", re.X)

        texts = [txt1, txt2, txt3, txt4, txt5, txt6, txt7, txt8, txt9, txt10]
        prompts = self.get_prompt_from_texts(texts, add_prompt)
        if len(prompts)!=0:
            p.prompt += ''.join(prompts)

        def shift_attention(text, distance):

            def inject_value(distance, match_obj):
                a = match_obj.group(1).split('~')
                l = len(a) - 1
                q1 = int(math.floor(distance * l))
                q2 = int(math.ceil(distance * l))
                return str(float(a[q1]) + ((float(a[q2]) - float(a[q1])) * (distance * l - q1)))

            res = re.sub(re_attention_span, lambda match_obj: inject_value(distance, match_obj), text)
            return res

        initial_info = None
        images = []
        dists = []
        lead_inout = int(lead_inout)
        tgt_w, tgt_h = round(p.width * upscale_ratio), round(p.height * upscale_ratio)
        save_video = video_fps > 0
        ssim_stats = {}
        ssim_stats_new = {}

        if not save_video and not show_images:
            print(f"Nothing to do. You should save the results as a video or show the generated images.")
            return Processed(p, images, p.seed)

        if save_video:
            import numpy as np
            try:
                import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            except ImportError:
                print(f"moviepy python module not installed. Will not be able to generate video.")
                return Processed(p, images, p.seed)

        # Custom folder for saving images/animations
        shift_path = os.path.join(p.outpath_samples, "shift")
        os.makedirs(shift_path, exist_ok=True)
        shift_number = Script.get_next_sequence_number(shift_path)
        shift_path = os.path.join(shift_path, f"{shift_number:05}")
        p.outpath_samples = shift_path
        if save_video: os.makedirs(shift_path, exist_ok=True)

        # Force Batch Count and Batch Size to 1.
        p.n_iter = 1
        p.batch_size = 1

        # Make sure seed is fixed
        fix_seed(p)

        initial_prompt = p.prompt
        initial_negative_prompt = p.negative_prompt
        initial_seed = p.seed

        # Kludge for seed travel
        p.subseed = p.seed

        # Split prompt and generate list of prompts
        promptlist = re.split("(THEN\(seed=[0-9]*\)|THEN)", p.prompt) + [None]
        negative_promptlist = re.split("(THEN\(seed=[0-9]*\)|THEN)", p.negative_prompt) + [None]

        # Build new list
        prompts = []
        while len(promptlist) or len(negative_promptlist):
            prompt, subseed, negprompt, negsubseed = (None, None, None, None)
            if len(promptlist):
                prompt = promptlist.pop(0).strip()
                subseed = promptlist.pop(0)
                if subseed:
                    s = re.sub("THEN\(seed=([0-9]*)\)", "\\1", subseed)
                    subseed = int(s) if s.isdigit() else None
            if len(negative_promptlist):
                negprompt = negative_promptlist.pop(0).strip()
                negsubseed = negative_promptlist.pop(0)
                if negsubseed:
                    s = re.sub("THEN\(seed=([0-9]*)\)", "\\1", negsubseed)
                    negsubseed = int(s) if s.isdigit() else None
            if not subseed:
                subseed = negsubseed
            # if subseed and len(prompts):
            #    prompts[-1] = (prompts[-1][0], prompts[-1][1], subseed)
            prompts += [(prompt, negprompt, subseed)]

        # Set generation helpers
        total_images = int(steps) * len(prompts)
        state.job_count = total_images
        print(f"Generating {total_images} images.")

        # Generate prompt_images and add to images (the big list)
        prompt = p.prompt
        negprompt = p.negative_prompt
        seed = p.seed
        subseed = p.subseed
        for new_prompt, new_negprompt, new_subseed in prompts:
            if new_prompt:
                prompt = new_prompt
            if new_negprompt:
                negprompt = new_negprompt
            if new_subseed:
                subseed = new_subseed

            p.seed = seed
            p.subseed = subseed

            # Frames for the current prompt pair
            prompt_images = []
            dists = []

            # Empty prompt
            if not new_prompt and not new_negprompt:
                print("NO PROMPT")
                break

            # DEBUG
            print(f"Shifting prompt:\n+ {prompt}\n- {negprompt}\nSeeds: {int(seed)}/{int(subseed)}")

            ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            # Generate the steps
            distances = curve_steps(curve, curvestr, int(steps))
            for i in range(int(steps) + 1):
                if state.interrupted:
                    break

                # distance = float(i / int(steps))
                distance = distances[i]
                p.prompt = shift_attention(prompt, distance)
                p.negative_prompt = shift_attention(negprompt, distance)
                p.subseed_strength = distance

                proc = process_images(p)

                # Checking SSIM minimum
                if len(prompt_images)>0:
                    transform = transforms.Compose([transforms.ToTensor()])
                    a = transform(prompt_images[-1].convert('RGB')).unsqueeze(0)
                    b = transform(proc.images[0].convert('RGB')).unsqueeze(0)
                    d = ssim(a, b)
                    trial=0
                    trials = []
                    trials_dist = []
                    while d<ssim_min:
                        if trial<5:
                            print("The generated image is way different! Regenerating..")
                            p.subseed = p.seed+ random.randint(0,2**32-p.seed)
                            proc = process_images(p)
                            trials.append(proc)
                            # a = transform(prompt_images[-1].convert('RGB')).unsqueeze(0)
                            b = transform(proc.images[0].convert('RGB')).unsqueeze(0)
                            d = ssim(a, b)
                            trials_dist.append(d)
                            trial+=1
                        else:
                            print("Even though the generated image is way different, it failed to regenerate, so just stick on the best image")
                            best_idx = trials_dist.index(max(trials_dist))
                            proc = trials[best_idx]
                            break
                        p.subseed = p.seed
                if initial_info is None:
                    initial_info = proc.info

                prompt_images += [proc.images[0]]
                dists += [distance]


            # SSIM
            if ssim_diff > 0:
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                if ssim_ccrop == 0:
                    transform = transforms.Compose([transforms.ToTensor()])
                else:
                    transform = transforms.Compose(
                        [transforms.CenterCrop((tgt_h * (ssim_ccrop / 100), tgt_w * (ssim_ccrop / 100))),
                         transforms.ToTensor()])

                # transform = transforms.Compose([transforms.ToTensor()])

                check = True
                skip_count = 0
                not_better = 0
                skip_ssim_min = 1.0
                min_step = 1.0

                done = 0
                while (check):
                    if state.interrupted:
                        break
                    check = False
                    for i in range(done, len(prompt_images) - 1):
                        # Check distance between i and i+1
                        a = transform(prompt_images[i].convert('RGB')).unsqueeze(0)
                        b = transform(prompt_images[i+1].convert('RGB')).unsqueeze(0)
                        d = ssim(a, b)


                        if d < ssim_diff and (dists[i + 1] - dists[i]) > substep_min:
                            # FIXME debug output
                            print(f"SSIM: {dists[i]} <-> {dists[i + 1]} = ({dists[i + 1] - dists[i]}) {d}")

                            # Add image and run check again
                            check = True

                            # ?? heuristic, distance moving by half.. subseed moving?
                            new_dist = (dists[i] + dists[i + 1]) / 2.0

                            p.prompt = shift_attention(prompt, new_dist)
                            p.negative_prompt = shift_attention(negprompt, new_dist)
                            p.subseed_strength = new_dist

                            # SSIM stats for the new image
                            # ssim_stats_new[(dists[i], dists[i + 1])] = d

                            print(f"Process: {new_dist}")
                            proc = process_images(p)

                            if initial_info is None:
                                initial_info = proc.info

                            # upscale - copied from https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel
                            if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
                                image = resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)
                            else:
                                image = proc.images[0]

                            # Check if this was an improvment
                            c = transform(image.convert('RGB')).unsqueeze(0)
                            d2 = ssim(a, c)

                            # if d2 > d or d2 < ssim_diff * ssim_diff_min / 100.0:
                            if d2 > d:
                                # Keep image if it is improvment or hasn't reached desired min ssim_diff
                                prompt_images.insert(i + 1, image)
                                dists.insert(i + 1, new_dist)
                                ssim_stats_new[(dists[i], dists[i + 1])] = d

                            else:
                                print(f"Did not find improvment: {d2} < {d} ({d - d2}) Taking shortcut.")
                                not_better += 1
                                done = i + 1
                            break;

                        else:
                            # DEBUG
                            if d > ssim_diff:
                                if i > done:
                                    print(f"Done: {dists[i + 1] * 100}% ({d}) {len(dists)} frames.   ")
                            else:
                                print(f"Reached minimum step limit @{dists[i]} (Skipping) SSIM = {d}   ")
                                if skip_ssim_min > d:
                                    skip_ssim_min = d
                                skip_count += 1
                            done = i
                            ssim_stats[(dists[i], dists[i + 1])] = d
                # DEBUG
                print("SSIM done!")
                if skip_count > 0:
                    print(
                        f"Minimum step limits reached: {skip_count} Worst: {skip_ssim_min} No improvment: {not_better}")




            # We should have reached the subseed if we were seed traveling
            seed = subseed

            # End of prompt_image loop
            images += prompt_images


        # Upscaling
        if upscale_meth != 'None' and upscale_ratio != 1.0 and upscale_ratio != 0.0:
            for idx, ori in enumerate(images):
                print(f"{idx}th image up-scaling")
                # image = resize_image(0, ori, tgt_w, tgt_h, upscaler_name=upscale_meth)
                images[idx] = run_bsr(ori)
            for idx, img in enumerate(images):
                im = Image.fromarray(img)
                im.save(os.path.join(p.outpath_samples, str(idx) + '.png'))


        if save_video:
            # frames = [np.asarray(images[0])] * lead_inout + [np.asarray(t) for t in images] + [
            #     np.asarray(images[-1])] * lead_inout
            frames = [np.asarray(t) for t in images]
            clip = ImageSequenceClip.ImageSequenceClip(frames, fps=video_fps)
            filename = f"no-rife-{shift_number:05}.mp4"
            clip.write_videofile(os.path.join(shift_path, filename), verbose=False, logger=None)



        # RIFE (from https://github.com/vladmandic/rife)
        if rife_passes:
            rifemodel = None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            count = 0

            w, h = tgt_w, tgt_h
            scale = 1.0
            fp16 = False

            tmp = max(128, int(128 / scale))
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)

            def rifeload(model_path: str = os.path.dirname(os.path.abspath(__file__)) + '/rife/flownet-v46.pkl',
                         fp16: bool = False):
                global rifemodel  # pylint: disable=global-statement
                torch.set_grad_enabled(False)
                if torch.cuda.is_available():
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    if fp16:
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
                rifemodel = Model()
                rifemodel.load_model(model_path, -1)
                rifemodel.eval()
                rifemodel.device()

            def execute(I0, I1, n):
                global rifemodel  # pylint: disable=global-statement
                if rifemodel.version >= 3.9:
                    res = []
                    for i in range(n):
                        res.append(rifemodel.inference(I0, I1, (i + 1) * 1. / (n + 1), scale))
                    return res
                else:
                    middle = rifemodel.inference(I0, I1, scale)
                    if n == 1:
                        return [middle]
                    first_half = execute(I0, middle, n=n // 2)
                    second_half = execute(middle, I1, n=n // 2)
                    if n % 2:
                        return [*first_half, middle, *second_half]
                    else:
                        return [*first_half, *second_half]

            def pad(img):
                return F.pad(img, padding).half() if fp16 else F.pad(img, padding)

            rife_images = frames

            for i in range(int(rife_passes)):
                print(f"RIFE pass {i + 1}")
                if rifemodel is None:
                    rifeload()
                print('Interpolating', len(rife_images), 'images')
                frame = rife_images[0]
                buffer = []

                I1 = pad(torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(
                    0).float() / 255.)
                for frame in rife_images:
                    I0 = I1
                    I1 = pad(torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(
                        0).float() / 255.)
                    output = execute(I0, I1, 1)
                    for mid in output:
                        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                        buffer.append(np.asarray(mid[:h, :w]))
                    if not rife_drop:
                        buffer.append(np.asarray(frame))
                rife_images = buffer

            frames = [np.asarray(rife_images[0])] * lead_inout + [np.asarray(t) for t in rife_images] + [
                np.asarray(rife_images[-1])] * lead_inout
            clip = ImageSequenceClip.ImageSequenceClip(frames, fps=video_fps)
            filename = f"shift-rife-{shift_number:05}.mp4"
            clip.write_videofile(os.path.join(shift_path, filename), verbose=False, logger=None)
        # RIFE end

        if order_frames:
            if rife_passes:
                images = frames

                for idx, img in enumerate(images):
                    im = Image.fromarray(img)
                    im.save(os.path.join(p.outpath_samples, str(idx) + '.png'))
            else:
                for idx, img in enumerate(images):
                    img.save(os.path.join(p.outpath_samples, str(idx) + '.png'))

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        end = time.time()
        sec = (end - start)
        result_list = str(datetime.timedelta(seconds=sec)).split(".")
        time_took = result_list[0]
        print(f"Total Time consumed: {time_took}")

        # SSIM-stats
        if save_stats and ssim_diff > 0:
            # Create scatter plot
            x = []
            y = []
            for i in ssim_stats_new:
                s = i[1] - i[0]
                if s > 0:
                    x.append(s)  # step distance
                    y.append(ssim_stats_new[i])  # ssim
            plt.scatter(x, y, s=1, color='#ffa600')
            x = []
            y = []
            for i in ssim_stats:
                s = i[1] - i[0]
                if s > 0:
                    x.append(s)  # step distance
                    y.append(ssim_stats[i])  # ssim
            plt.scatter(x, y, s=1, color='#003f5c')
            plt.axvline(substep_min)
            plt.axhline(ssim_diff)

            plt.xscale('log')
            plt.title('SSIM scatter plot')
            plt.xlabel('Step distance')
            plt.ylabel('SSIM')
            filename = f"ssim_scatter-{shift_number:05}.svg"
            plt.savefig(os.path.join(shift_path, filename))
            plt.close()

        # Save settings and other information
        if save_stats:
            D = []

            # Settings
            D.extend(['Prompt:\n', initial_prompt, '\n'])
            D.extend(['Negative prompt:\n', initial_negative_prompt, '\n'])
            D.append('\n')
            D.extend(['Checkpoint: ', shared.sd_model.sd_checkpoint_info.name,'\n'])
            D.extend(['Width: ', str(p.width), '\n'])
            D.extend(['Height: ', str(p.height), '\n'])
            D.extend(['Sampler: ', p.sampler_name, '\n'])
            D.extend(['Steps: ', str(p.steps), '\n'])
            D.extend(['CFG scale: ', str(p.cfg_scale), '\n'])
            D.extend(['Seed: ', str(initial_seed), '\n'])
            D.append('\n')
            D.append('-- Interpolation settings ------------\n')
            # Shift Attention Settings
            D.extend(['Steps: ', str(int(steps)), '\n'])
            D.extend(['FPS: ', str(video_fps), '\n'])
            D.extend(['Lead in/out: ', str(int(lead_inout)), '\n'])
            D.extend(['SSIM minimum: ', str(ssim_min), '\n'])
            D.extend(['SSIM threshold: ', str(ssim_diff), '\n'])
            # D.extend(['SSIM CenterCrop%: ', str(ssim_ccrop), '\n'])
            D.extend(['RIFE passes: ', str(int(rife_passes)), '\n'])
            D.extend(['Drop original frames: ', str(rife_drop), '\n'])
            D.extend(['Upscaler: ', upscale_meth, '\n'])
            D.extend(['Upscale ratio: ', str(upscale_ratio), '\n'])
            D.extend(['Strength Curve Type: ', str(curve), '\n'])
            D.extend(['Strength: ', str(curvestr), '\n'])
            # D.extend(['SSIM min substep: ', str(substep_min), '\n'])
            # D.extend(['SSIM min threshold: ', str(ssim_diff_min), '\n'])
            D.append('---------------------------------------\n')
            # Generation stats
            if ssim_diff:
                D.append(
                    f"Stats: Skip count: {skip_count} Worst: {skip_ssim_min} No improvment: {not_better} Min. step: {min_step}\n")
            D.append(f"Total number of Frames: {len(images)}\n")
            D.append('---------------------------------------\n')
            D.append(f'Total Time took: {str(time_took)}')


            filename = f"generation-info-{shift_number:05}.txt"
            file = open(os.path.join(shift_path, filename), 'w')
            file.writelines(D)
            file.close()
        return processed

    def describe(self):
        return "Shift attention in a range of images."
