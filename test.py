question = data["question"]
passages = data["passages"]
answer = data["answer"]
def get_pred(model, psgs):
    text = predict(model, tokenizer, generation_config,question, with_cot=args.with_cot, passages=psgs)
    pred = {"test_id": test_id, "question": question, "answer": answer, "text": text}
    pred.update(evaluate(text, answer, args.with_cot))
    return pred
all_deltas = []
for passage in passages:
    tokens = tokenizer(passage,padding=True,truncation=True,return_tensors="pt",max_length=3000).to(model.device)
    with torch.no_grad():
        output = model(tokens.input_ids, output_hidden_states=True)
        input_embeds = output.hidden_states[-1][:,-1,:]
        outputs = projector(input_embeds)
        all_deltas.append(outputs)
merged_deltas = {}
for key in all_deltas[0].keys():
    merged_deltas[key] = torch.stack([delta[key] for delta in all_deltas]).mean(dim=0)
delta_inject(model, merged_deltas)
ret.append(get_pred(model, psgs=None if args.inference_method == "dyprag" else passages))
delta_remove(model, merged_deltas)