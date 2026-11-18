import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import run_compute_per_instance_dpo_loss as compute_per_instance_dpo_loss
from .common import FIXTURES_PATH


def test_per_instance_dpo_loss():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2")
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    dpo_inputs = dict(
        lm=model,
        lm_ref=model_ref,
        tokenizer=tokenizer,
        beta=0.5,
        prompt=prompt,
        response_chosen=good_response,
        response_rejected=bad_response,
    )

    loss = compute_per_instance_dpo_loss(**dpo_inputs)
    assert torch.isclose(loss, torch.tensor(0.5785), atol=1e-4)

    policy_before = [parameter.detach().clone() for parameter in model.parameters()]
    reference_before = [parameter.detach().clone() for parameter in model_ref.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        loss_after_step = compute_per_instance_dpo_loss(**dpo_inputs)
    assert any(not torch.equal(parameter, before) for parameter, before in zip(model.parameters(), policy_before))
    for parameter, before in zip(model_ref.parameters(), reference_before):
        assert parameter.grad is None
        assert torch.equal(parameter, before)
    assert loss_after_step < loss.detach(), f"DPO loss: {loss.item()} -> {loss_after_step.item()}"
