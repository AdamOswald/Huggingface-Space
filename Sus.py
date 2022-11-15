examples = ["a <cat-toy> in <madhubani-art> style", "a <line-art> style mecha robot", "a piano being played by <bonzi>", "Candid photo of <cheburashka>, high resolution photo, trending on artstation, interior design"]

block = gr.Blocks(css=css)

examples = [
    [
        'Goku'
    ],
    [
        'Mikasa Ackerman'
    ],
    [
        'Saber'
    ],
]

with block as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 720px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                  <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAkCAYAAADhAJiYAAANeElEQVRYhY1Ye4xc1X3+zjn33Oc8dmdmZ5+zttfrZ/wIDmAgJgkWoTZKW1WQRG1T0aZNG7VSK7XiL1SlNIqaKEpQUlWqkj5CCCmUpECBGMLDDWAwsbF5LTb78u7s7ni9u/O+M/d9T3XuGjsRsslIVzNXd+653+/3+87v932XAEBloSK/QAiBqmlwux2YKQsijGBRDg6KtchBEAYQsUA2lYJCGDzHkTchpgSxENBVjlhVwCMMNpuN/cTke1zX+5SqqHcxpiwICBQLfbjSRz5fueLVK3yGSyNoVWupQAjXtMwwDCPEnCH0/Gx9ZuHzlVNnDrfemrzBjsOBQ9+6G4ypLUJIU1H4b7T+BwEJGQfATR0II5VTDgIa6TEiAyZ5+AcPfOexh39yG6Wk+nufvfPRnbt3zdJaZ3jpsRe/vHzynZ1RpwvEEXipiPpKDdC1duSHQkZvaDpgWb85ICHEkGYYX6ktVLZP/PgpS1NVK+q4oJEIcltGW9NuXfmX++673nddmV7yT/d+9aZUOo20D2wTaVyTHoBmGYjCEFTXIQSBzrg5W5lTVVXF5rEt8GwXWkq/OiDXddcBAYYl4j+jkaBzDx6BGgtQRmETgVNhDbNxG6CAaVmI4zi5LqIIdSrwKmooey5uNUaQEQCxVHBTQ+D7ztjGMd/zPNSqVVIaGxUfmiFd1xNCxXG8GIXhXK7YN5ZOp0B8F5QpeClcwRT1oAgVmryJKYjiCLEETMg6sUWM86GLo90l3MqKKI2PoK/Yh6WFpTVFUboTExM48rMjoqenB3//1a9cHRClLMkPo8xzOt2yavAxUioAk4tYpC7KUQcm4eCqkvBLgjc5h4xaZkqeJ/eDYTUMsER97Nq5GSvVKjzfOwKCaHh0BKlUCjfcdONVOUTXAREEYZhsa9/zv840Hu/90zsQEIJzUQcxY9AVDbqmgwjANAxkUilIGEnfiEVyLZvOJMFNpQUwVMBCecmxUunvqZqOQl8RX/6rv8R111//4YAURYGIY3S7XQSB//PVyvIb6oYB8Fs+ioprw2IadFNHEAQyYjiOg1q9ngQhORRFIVzHTa4pjGHN7cBxPFiW9bLn+bOCAkRXYFhmEkBttXZ1QEIIpNNpDA+PoFQaFZxrT2pye+7eiLXYQxTHaDTrqNVroIQiCELY3S66rovenhRUlSX/abba8KMQzUYTZ995B3atxjL5zFhEYnhRiDDhXUxyfbmrA3oflOSCPPL53Hfatn3Ktm2EUQzHc+F6HkAoPrp7M0yNwXNduG6A0ZEh9PWmYZoqZD3DKEJMgcWjv8TUNx842Fo8/6Su8BuNECT0AwRR9OElex9QGAZJ+lVVrW3etOlOhNGK57lJaTw/QDaTwrV7tiGbkeXzICKCnoyJkcF+DA/mkU4ZcF0PoePijbMTsM/M4+S3H9hBg+ggjyECPyBhGJKrlexSY5SZcbpdeN2k6YFr6vLp1085oRfAjtoglCGXS2P39s1otjuwvQhTk+cx2J9DT2EjzkydQ7XVgUMoNpSGMTc9gwUaYdNA/6rwwlmPERiGIbiigDEGu2kjlU1dGRCSfmRA5SqYoqBRr+89e+Zs6eCh27DrumvAEePtk6/i/od/ii1jY/jSZ2/HDx8+gly6F6+9/BI27LkGnxvfjc0bt+H4qddx+sRJNA7sws13f/GPXM97ptu2kc/lZCWSxrg+oK6SIUSynwAKodAUjna9UXIdh87Pz+HGkc3wm3Xc+ZnDmHr9NezYsxM7hooYyOehGxo2Dd6GgVIJ9mITJ75xP36xfBYRARYrS+janW6ssk90bLutMHY6l1sntN2x0a046B8qfhCQnMSMUAgRJ2SNIfLVtbV7ZF+aK8/h9fn/RamQw/AXDmLf/i9CBEDMFHxk07hsExAiAuMU1U4ZoRDI9faCVVtSnoiZmcl7iyPDO8NY9E9Pz74wPOx8P5PJPCQ7+xUzpGla0v5FHMmo+lSF/1uj0dibzWTRqtcQ5TIY2L0Lqw7FcN8A4HqAWNdAXI4drqDVrGOm7SH3x4ex50IF8z97GsX+XrKhaN3CmIuW38XQ4NBBK2V93PXdUY2r/y6AqmM7MFLGB3cZWVcePQrnD4CQ3yGguPbaffC9EGN33IrhLxzG9Mwy3jv9LsI4BhhB6LkgjKDVqGLx7bdBRYz/eOJR/OeDP8LY+AjGx8dQGCgilzKxtdSLtMlR6Cto1bmFb0RxdCqO4z+UG8jteL8OqF6voVarysZXKhaLt8mZMzDQixs/vhuazvFfDzyIr//j11AYHECnXMHkiRM4vziPkEVYnJ/F6nuTyIKiY3uozM0j39uDHTvHcd3+a0A0LVGUhAmoGgfCEDP/8xxMVRs1Let79Xr905TSS6DWZ1kkkqPZbB5gTCFUYSiNFHDgwE3Ytn0jOnYThmki298H3dJQJIA9NY3zJ07BmysjG4WohwRdQdFqNHDzJ2/A4ds/jX0f24NgdQXuygV0601wM42FN88itj3opoF0OmUKIZ6oVCp3qCq/zCFDN8Aova46X/k220ihcQ6ma+jN9eB3f/sw5mcXUezvRybfg1ozDdddQ95KgQkColKsNlsI+rdDrU0ixTl+69BBjJRG0G7WcPzENNpND0ZvFsNxFsefO4pP/t1d4KYJ4QYoFoua7/sPNZvN3wfwkyRDRGdKtbZ678z/vapTzpDOpnF+tY533pzA2GABmXQGuVwehqKgd8NmLGl5tL0ITOVodAOs8gLGdu6FY9voZQYsKAgdB0uzqy/0lHb8rZIvHnOJ+sbZibPtkf17QXUDfscFoesUppQq5YWFQ5dJTcg26PxmZ3kNtck5rNYb6HgMq80Y52wKP4wxMNCH2POgayryGzbjzIVlPPSjJ3Hs+DFwS4NqcSwuVSA6Hupvl9FcWcW505Mr45s33rd9+/iBsbHSHYZmOJpioLJYQadtJ+Mo6doykJ6e5iVASwuV4VQ6E33ir+/CY9//IZ596DFUl1bAVQNyAfghTDfAyR8/jmxPFpNPH8PS8Rex4zYLGzf5ePkf/hnvPvsiVirn0avpmHvuOGpzZRCnW4pZhFxfAavV2t2Ua0Wu6ggjgXOz86jX6skwdlyn5nn+dy9xKJWyLtTX6oGRtrD3c59BfXkNruMjlSUoz5WhpC3A87D8zKt4c6iIyn+/gMGtWey7ZRRTzy0hfSHCU/d8E1Osjv3ZAmoz59EuV+R439Ou1gZbrrez1Wh9qbe3kGgvXctAUSladhvBUoC16trC+Jbx+V8BlF7k3G07jldQqApFt+CHnaSDywWGRkswM1m0Gx1M/+ujUD0Byywg6ggwTQGzVKR9H5YXIsNVLLsOXnnqJFRLS9cfefJrpRv2btJ1jcUihBAKDMPCyKahRBAulMswTfOcetG3XexD9TrnvJ6RnbndThShlKIRBHryeYxu2Ijxm69HvG0QVaeJ890G3IyHlK5i064i3D5grd1Bn6eg3u3AZTEWp5YweeJdHHvk8T+pLC5+ygvcZDSta6YARLo9Kg8qTcNr1Vr9MiBd1+Iois7J4SpnDOccpmWgv78PUpyPbd0KlxJ87C8+jwvwUO7WEeZUxHEEw1KQ3qCi03WRjTiCOE60ldytmmFBb3poTcyg1mzA83wQIhAnsjeC7/uJ5FVVXsbF2ZYAyudzEukjvh9IzQKFK1AUBkPXEiuzZdt2KDGwPD0LBgquUwyMmPADH2EUorSvH4rKQCi5pDoJoUmGmR9h7pmXYPohXDmIgUQ1hmEI+bxYalpCqpIalwC12x3psZ7xPLcuPZf0WlxlCEJfWiMUh4agCIHpo8fg2B1ksikMjhQgNUYcxShtzUJLkXUJc3EuJr5NCCiqBu4GcN6aRmRLHe4gFlEijaMwllo8tm17ttFoXO7U2WwPmELcyfdmXg3D8HZZVzlc5Q2y7opCcOrxn+PC6UkoRIGSYjCzGoSUx1TATGugGYawG4JBgEQxOKGQI0gWQoQxJp44ikHHR3TopiQ7MrNS3hAg2Lp1a1KVX5v2Ugpls5l7/MD1OGfrbjQGFFXF8sQkpp44Ck2OGMZgZDSoKZb8ljpK9hah80SUMSpfKnBoCgUlAgoFOGXQKMeF54+DrjXQm88jcL3EaDLGTrRa7dla9VdILckspWWhkO8IERFJPMNQpQNMan7m6V8gtDugnIJxgqglcPrpMipLNsrLHUycXUaPqiOr6DC5nlglkSxOEgXKGIWqKzAJwfnnfwlOCOxOVw5zab94tidjSht2qWTiIsMZo7MjpZEfLM4v/rn04IqmoLlYwewLr0DXVRApg5iCYM3D8996A+kNBtQsR2etg6gGKDpN+HHZUtGE6PQi0Zmmov7WGSyfncHg3o9I/rmGYX438MPORam9Doip7JKy3jS24W8Cz09FfvAHUsm9fuw1BK02VEtNbLRcnDECwjncig9nQaZdZmE9MBmbFPCyjymUrZNcqlEJklKoATD902eRHR2BZhhvraxVH1pZq66/e3kfUOiHlzUtU9z+gf77gzA8tLa6gnOvnIShKrBUA0G83j8YYVLBJkAEXV9IBngxSGiMQ2U82W7Sor//gkKec01D891ZrM6X66Vd2x6W7wUu+Q8A/w+34pKwK2kanwAAAABJRU5ErkJggg==" />
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Image Generation Test
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Test My Concept
              </p>
            </div>
            <div class="finetuned-diffusion-div">
               <div>
                <h1>Finetuned Diffusion</h1>
               </div>
                <p>
                Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: <br>
                 <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>, <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spiderverse</a>, <a href="https://huggingface.co/nitrosocke/modern-disney-diffusion">Modern Disney</a>, <a href="https://huggingface.co/hakurei/waifu-diffusion">Waifu</a>, <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokemon</a>, <a href="https://huggingface.co/yuk/fuyuko-waifu-diffusion">Fuyuko Waifu</a>, <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony</a>, <a href="https://huggingface.co/sd-dreambooth-library/herge-style">Hergé (Tintin)</a>, <a href="https://huggingface.co/nousr/robo-diffusion">Robo</a>, <a href="https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion">Cyberpunk Anime</a> + any other custom Diffusers 🧨 SD model hosted on HuggingFace 🤗.
                 </p>
                 <p>Don't want to wait in queue? <a href="https://colab.research.google.com/gist/qunash/42112fb104509c24fd3aa6d1c11dd6e0/copy-of-fine-tuned-diffusion-gradio.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a></p>
                  Running on <b>{device}</b>
              </p>
            </div>
            <div class="acknowledgments">
                     <p><h4>LICENSE</h4>
 The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                     <p><h4>Biases and content acknowledgment</h4>
 Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
            </div>
        """
    )
    
with gr.Blocks(css=css) as demo:

    gr.Markdown('''
      👇 Buy me a coffee if you like ♥ this project~ 👇 Running this server costs me $100 per week, any help would be much appreciated!
      [![Buy me a coffee](https://badgen.net/badge/icon/Buy%20Me%20A%20Coffee?icon=buymeacoffee&label)](https://www.buymeacoffee.com/dgspitzer)
    ''')
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=models, value=models[0])
            prompt = gr.Textbox(label="Prompt", placeholder="{} is added automatically".format(prompt_prefixes[current_model]), elem_id="input-prompt")
            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="Steps", value=27, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            image_out = gr.Image(height=512, type="filepath", elem_id="output-img")

    with gr.Column(elem_id="col-container"):
        with gr.Group(elem_id="share-btn-container"):
          community_icon = gr.HTML(community_icon_html, visible=False)
          loading_icon = gr.HTML(loading_icon_html, visible=False)
          share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
          
    model.change(on_model_change, inputs=model, outputs=[])
    run.click(inference, inputs=[prompt, guidance, steps], outputs=[image_out, share_button, community_icon, loading_icon, prompt])
    
    share_button.click(None, [], [], _js=share_js)
       
    gr.Examples([
        ["portrait of anime girl", 7.5, 27],
        ["a beautiful perfect face girl, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece by studio ghibli, 8k, sharp high quality anime, artstation", 7.5, 27],
        ["cyberpunk city landscape with fancy car", 7.5, 27],
        ["portrait of liu yifei girl, soldier working in a cyberpunk city, cleavage, intricate, 8k, highly detailed, digital painting, intense, sharp focus", 7.5, 27],
        ["portrait of a muscular beard male in dgs illustration style, half-body, holding robot arms, strong chest", 7.5, 27],
    ], [prompt, guidance, steps], image_out, inference_example, cache_examples=torch.cuda.is_available())
    gr.Markdown('''
      Models and Space by [@DGSpitzer](https://huggingface.co/DGSpitzer)❤️<br>
      [![Twitter Follow](https://img.shields.io/twitter/follow/DGSpitzer?label=%40DGSpitzer&style=social)](https://twitter.com/DGSpitzer)
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=dgspitzer_DGS_Diffusion_Space)
      
      ![Model Views](https://visitor-badge.glitch.me/badge?page_id=Cyberpunk_Anime_Diffusion)
      
    ''')

with gr.Blocks(css=css) as demo:
  state = gr.Variable({
        'selected': -1
  })
  state = {}
  def update_state(i):
        global checkbox_states
        if(checkbox_states[i]):
          checkbox_states[i] = False
          state[i] = False
        else:
          state[i] = True
          checkbox_states[i] = True   

    with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
               
        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="generated_id").style(
            grid=[2], height="auto"
        )
        
        ex = gr.Examples(examples=examples, fn=infer, inputs=[text], outputs=gallery, cache_examples=True)
        ex.dataset.headers = [""]
        
        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)

       with gr.Row():
        with gr.Column():
          gr.Markdown(f"### Navigate the top 100 Textual-Inversion community trained concepts. Use 600+ from [The Library](https://huggingface.co/sd-concepts-library)")
          with gr.Row():
                  image_blocks = []
                  #for i, model in enumerate(models):
                  with gr.Box().style(border=None):
                    gr.HTML(assembleHTML(models))
                      #title_block(model["token"], model["id"])
                      #image_blocks.append(image_block(model["images"], model["concept_type"]))
        with gr.Column():
          with gr.Box():
                  with gr.Row(elem_id="prompt_area").style(mobile_collapse=False, equal_height=True):
                      text = gr.Textbox(
                          label="Enter your prompt", placeholder="Enter your prompt", show_label=False, max_lines=1, elem_id="prompt_input"
                      ).style(
                          border=(True, False, True, True),
                          rounded=(True, False, False, True),
                          container=False,
                          full_width=False,
                      )
                      btn = gr.Button("Run",elem_id="run_btn").style(
                          margin=False,
                          rounded=(False, True, True, False),
                          full_width=False,
                      )  
                  with gr.Row().style():
                      infer_outputs = gr.Gallery(show_label=False, elem_id="generated-gallery").style(grid=[2], height="512px")
                  with gr.Row():
                    gr.HTML("<p style=\"font-size: 95%;margin-top: .75em\">Prompting may not work as you are used to. <code>objects</code> may need the concept added at the end, <code>styles</code> may work better at the beginning. You can navigate on <a href='https://lexica.art'>lexica.art</a> to get inspired on prompts</p>")
                  with gr.Row():
                    gr.Examples(examples=examples, fn=infer_examples, inputs=[text], outputs=infer_outputs, cache_examples=True)
          with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
  checkbox_states = {}
  inputs = [text]
  btn.click(
        infer,
        inputs=inputs,
        outputs=[infer_outputs, community_icon, loading_icon, share_button]
    )
  share_button.click(
      None,
      [],
      [],
      _js=share_js,)       
       with gr.Blocks(css=css) as demo:

    with gr.Row():
        
        with gr.Group():
            model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
            custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", visible=False, interactive=True)
            
            with gr.Row():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="Enter prompt. Style applied automatically").style(container=False)
              generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))


            image_out = gr.Image(height=512)
            # gallery = gr.Gallery(
            #     label="Generated images", show_label=False, elem_id="gallery"
            # ).style(grid=[1], height="auto")

        with gr.Tab("Options"):
          with gr.Group():
            neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

            # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

            with gr.Row():
              guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
              steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=100, step=1)

            with gr.Row():
              width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
              height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

            seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

        with gr.Tab("Image to image"):
            with gr.Group():
              image = gr.Image(label="Image", height=256, tool="editor", type="pil")
              strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    model_name.change(lambda x: gr.update(visible = x == models[0].name), inputs=model_name, outputs=custom_model_path)
    custom_model_path.change(custom_model_changed, inputs=custom_model_path)
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt]
    prompt.submit(inference, inputs=inputs, outputs=image_out)
    generate.click(inference, inputs=inputs, outputs=image_out)

    ex = gr.Examples([
        [models[1].name, "jason bateman disassembling the demon core", 7.5, 50],
        [models[4].name, "portrait of dwayne johnson", 7.0, 75],
        [models[5].name, "portrait of a beautiful alyx vance half life", 10, 50],
        [models[6].name, "Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7.0, 45],
        [models[5].name, "fantasy portrait painting, digital art", 4.0, 30],
    ], [model_name, prompt, guidance, steps, seed], image_out, inference, cache_examples=not is_colab and torch.cuda.is_available())
    # ex.dataset.headers = [""]

    gr.Markdown('''
      Models by [@nitrosocke](https://huggingface.co/nitrosocke), [@Helixngc7293](https://twitter.com/DGSpitzer) and others. ❤️<br>
      Space by: [![Twitter Follow](https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social)](https://twitter.com/hahahahohohe)
  
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=anzorq.finetuned_diffusion)
    ''')
    block = gr.Blocks(css=css)

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
#        4,
#        45,
#        7.5,
#        1024,
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        'A mecha robot in a favela in expressionist style',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        'an insect robot preparing a delicious meal',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
#        4,
#        45,
#        7,
#        1024,
    ],
]


with block:

    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

        with gr.Row(elem_id="advanced-options"):
            gr.Markdown("Advanced settings are temporarily unavailable")
            samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(examples=examples, fn=infer, inputs=text, outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        ex.dataset.headers = [""]

        text.submit(infer, inputs=text, outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=text, outputs=[gallery], postprocess=False)
        
        advanced_button.click(
            None,
            [],
            text,
            _js="""
            () => {
                const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
                options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
            }""",
        )
        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )

     with gr.Row():
       with gr.Column():
           input_img = gr.Image(type="filepath", elem_id="input-img")
           with gr.Row():
             see_prompts = gr.Button("Feed in your image!")

       with gr.Column():
         img2text_output = gr.Textbox(
                                 label="Generated text prompt", 
                                 lines=4,
                                 elem_id="translated"
                             )
         with gr.Row():
             diffuse_btn = gr.Button(value="Diffuse it!")
       with gr.Column(elem_id="generated-gallery"):
         sd_output = gr.Gallery().style(grid=2, height="auto")
         with gr.Group(elem_id="share-btn-container"):
             community_icon = gr.HTML(community_icon_html, visible=False)
             loading_icon = gr.HTML(loading_icon_html, visible=False)
             share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

     see_prompts.click(get_prompts, 
                             inputs = input_img, 
                             outputs = [
                                 img2text_output
                             ])
     diffuse_btn.click(get_images, 
                           inputs = [
                               img2text_output
                               ], 
                           outputs = [sd_output, community_icon, loading_icon, share_button]
                           )
     share_button.click(None, [], [], _js=share_js)

if not is_colab:
  demo.queue(concurrency_count=4)
demo.launch(debug=is_colab, share=is_colab)
demo.queue(max_size=25).launch()
demo.queue()
demo.launch()
