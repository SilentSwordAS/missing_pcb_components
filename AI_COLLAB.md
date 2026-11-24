## AI Collaboration:
- Started the conversation with mentioning the detailed instructions, the rules/constraints to abide and what framework to use for creating this application(Streamlit).
- Prompted the AI to make necessary changes to how the bounding boxes was displayed to make sure that it was clear to the user which part they were labelling at the time.
- Used a few prompts to work on editing the training images, introducing missing components in them for testing.
- AI suggested a SSIM (Similarity Matching) based approach for the problem but the major issue with this was the lighting changes caused by absence of certain components made the similarity matching produce unexpected results. Hence, prompted the LLM to steer towards a more ML oriented approach with YOLO26.
- Modified the UI a bit more to ensure better user accessibility, compiled a list of alternative approaches towards the problem. Also, prompted for a better configuration of fine-tuning if possible. Finalised the YOLO26 with 50 epochs and 1024 resolution size and introduction of synthetic samples involving missing components during training.
