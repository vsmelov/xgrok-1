import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import time

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"

app = FastAPI()


# Define the data model for the request
class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 100
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.5


model_lock = threading.Lock()


def load_model():
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()
    return gen


def get_reply(gen, inp):
    reply = sample_from_model(gen, inp, max_len=100, temperature=0.01)
    print(f"Output for prompt: {inp}", reply)
    return reply


process_scope = {}
logging.info('Loading model...')
process_scope['inference_runner'] = load_model()


@app.post("/inference/")
def do_inference(request: TextRequest):
    logging.info(f"Received request: {request}")
    with model_lock:
        start_time = time.time()

        if 'inference_runner' not in process_scope:
            logging.info('Loading model...')
            process_scope['inference_runner'] = load_model()

        inference_runner = process_scope['inference_runner']

        try:
            # Assuming the model's inference method and other details
            output = sample_from_model(
                inference_runner,
                request.text,
                max_len=request.max_new_tokens,
                temperature=request.temperature,
            )
            response = inference_runner.tokenizer.decode(output)
            duration = time.time() - start_time
            logging.info(f'Inference took {duration:.2f} seconds')
            logging.info(f"Response: {response}")
            return {"response": response, "duration": duration}
        except Exception as e:
            logging.exception(f'Error occurred: {e}')
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
