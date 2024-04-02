# LLM Basics with huggingface transformers pipeline

## Installing necessary libraries

```!pip install transformers accelerate autoawq``` 

## Importing given libraries

``` python 
from transformers import {model type}, AutoTokenizer, pipeline
```

Enter the suitable model type for your required task in place of model type.


## Loading the pretrained model and tokenizer

``` python
model_name = "{enter your model path here}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
)


```

## Define a function to generate inference response

```python
def generate_response(prompt):

    prompt_template=f'''<s>[INST] {prompt} [/INST]
    '''

    generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        **generation_params
    )

    pipe_output = pipe(prompt_template)[0]['generated_text']

    return pipe_output
```

## A simple application of your choice

```python
print("Welcome to Local GPT, you can get your questions answered here. Type 'exit()' to exit")

while True:

    user_input = str(input("User: "))

    if user_input == "exit()":
      break
    else:
      output = generate_response(user_input)

      if output.startswith("<s>[INST]"):
        index = output.find("[/INST]")
        output = output[index+12:]
      print("AI: ",output)
      print("**"*100)
```

Here, this code is done as a simple terminal code, but this can be enhanced further by using libraries like streamlit or grado.
