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
                  <img src="data:image/png;base64,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" />
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Image Generation Test
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Test My Concept
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
