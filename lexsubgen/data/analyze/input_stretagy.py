for word in proposed_words:
    proposed_words[word] = proposed_words[word] + alpha * scores[word]  # alph=0.05


inputs_embeds_synonyms = None
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # 嵌入处理——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
            if noise_type == "GAUSSIAN":
                inputs_embeds[0][word_index] = self.noise(inputs_embeds[0][word_index])
            elif noise_type == "GLOSS":
                if input_ids_synonyms is not None:
                    if len(input_ids_synonyms) > 0:
                        for index_id in range(0, len(input_ids_synonyms)):
                            if inputs_embeds_synonyms is None:
                                inputs_embeds_synonyms = torch.sum(
                                    # 将同义词的 token id 转换为嵌入，并对嵌入求和，然后除以该同义词 token 数量，从而得到该同义词的【平均嵌入】。
                                    self.word_embeddings(torch.tensor(input_ids_synonyms[index_id]).to(device)),
                                    dim=0) / len(input_ids_synonyms[index_id])
                            else:
                                inputs_embeds_synonyms = inputs_embeds_synonyms + torch.sum(
                                    self.word_embeddings(torch.tensor(input_ids_synonyms[index_id]).to(device)),
                                    dim=0) / len(input_ids_synonyms[index_id])

                        sum_synonym = inputs_embeds_synonyms / len(input_ids_synonyms)

                        temp = lambda_variable * inputs_embeds[0][word_index] + (1 - lambda_variable) * sum_synonym
                        inputs_embeds[0][word_index] = temp.squeeze(0)
                    else:
                        print("synonyms not found")
                        exit(5)
            elif noise_type == "DROPOUT":
                if not self.dropout_embedding.training:
                    self.dropout_embedding.train()
                inputs_embeds[0][word_index] = self.dropout_embedding(inputs_embeds[0][word_index])
            else:
                pass
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings  # 三个编码的相加
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings