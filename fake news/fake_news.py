class fn():

    def __init__(self, text):
        self.text = re.sub("(<span class='quot[0-9]'>|\n\r\n|</span>|<br/>|<br />|([^0-9가-힣A-Za-z. ]))","",text)

    def __str__(self):
        return f"해당 클래스는 '{self.text}'라는 문장이 참인지, 거짓인지 판단합니다."

    def summarize(self):
        tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        model_sm = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        text = self.text.replace('\n', ' ')
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        if len(input_ids) > 1024:
            summary_text = '해당 기사 본문 내용이 너무 길어서 요약에 실패했습니다. 좀더 적은 양의 본문 내용을 뽑아주세요'
        else:
            summary_ids = model_sm.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
            summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

        return summary_text

    def sts(self):
        query_embedding = model_sts.encode(self.text)
        cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)
        num_top_results = sum(top_results[0] > 0.45)

        cos_contents = []
        label_contents = []

        if num_top_results != 0:
            for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                if score <= 0.5: break
                label = 'True' if data['label'][int(idx)] == 1 else 'False'
                cos_contents.append(content[idx])
                label_contents.append(label)
            return cos_contents, label_contents

        elif num_top_results == 0:
            return "입력 문장과 유사한 문장이 없으므로 확인이 불가능합니다"

    def nli(self, con_text):
        classifier = pipeline("text-classification", model="Huffon/klue-roberta-base-nli", return_all_scores=True)
        tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")
        results = classifier(f'{self.text} {tokenizer.sep_token} {con_text}')[0]
        return results

    def fact(self):
        if self.sts() == "입력 문장과 유사한 문장이 없으므로 확인이 불가능합니다":
            return "입력 문장과 유사한 문장이 없으므로 확인이 불가능합니다"

        else:
            sts_list = []
            for i in self.sts()[0]:
                sts_list.append(i)

            label_list = []
            for i in self.sts()[1]:
                label_list.append(i)

            nli_list = []
            for i in sts_list:
                nli_list.append(self.nli(i))

            fact_list = []

            for i in range(len(nli_list)):
                score = []
                for j in nli_list[i]:
                    score.append(j['score'])

                if score[0]>score[2]:
                    if label_list[i] == 'True':
                        fact_list.append('True')
                    else:
                        fact_list.append('False')

                else:
                    if label_list[i] == 'True':
                        fact_list.append('False')
                    else:
                        fact_list.append('True')

            return fact_list

    def result(self):
        if self.fact() == "입력 문장과 유사한 문장이 없으므로 확인이 불가능합니다":
            return "입력 문장과 유사한 문장이 없으므로 확인이 불가능합니다"
        else:
            if len(self.fact()) == 1:
                return f"입력 문장'{self.text}'에 대한 진위 여부는 '{self.fact()}'입니다."
            elif len(self.fact()) == 2:
                if self.fact() == ['True','False'] or self.fact() == ['False','True']:
                    return f"입력 문장'{self.text}'에 대한 판단을 유보합니다"
                else:
                    return f"입력 문장'{self.text}'에 대한 진위 여부는 '{self.fact()[0]}'입니다."

            else:
                t = 0
                f = 0
                for i in self.fact():
                    if i == 'True':
                        t += 1
                    else:
                        f += 1
                if t>f:
                    return f"입력 문장'{self.text}'에 대한 진위 여부는 'True'입니다."
                else:
                    return f"입력 문장'{self.text}'에 대한 진위 여부는 'False'입니다."