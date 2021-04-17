def init():











def ini2():

    # predict
    res_dict = {}
    predict_statistics = {}

    for run_id, model_path_list in run_id2model_state_paths.items():
        save_dir4run = os.path.join(save_res_dir, run_id)
        if test_config["save_res"] and not os.path.exists(save_dir4run):
            os.makedirs(save_dir4run)

        for model_state_path in model_path_list:
            # load model state
            rel_extractor.load_state_dict(torch.load(model_state_path))
            rel_extractor.eval()
            print("run_id: {}, model state {} loaded".format(run_id, model_state_path.split("/")[-1]))

            for file_name, short_data in test_data_dict.items():
                res_num = re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
                save_path = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num))

                if os.path.exists(save_path):
                    pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding="utf-8")]
                    print("{} already exists, load it directly!".format(save_path))
                else:
                    # predict
                    ori_test_data = ori_test_data_dict[file_name]
                    pred_sample_list = predict(short_data, ori_test_data)

                res_dict[save_path] = pred_sample_list
                predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
    pprint(predict_statistics)

    # %%

    # check
    for path, res in res_dict.items():
        for sample in tqdm(res, desc="check char span"):
            text = sample["text"]
            for rel in sample["relation_list"]:
                assert rel["subject"] == text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
                assert rel["object"] == text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]

    # save
    if test_config["save_res"]:
        for path, res in res_dict.items():
            with open(path, "w", encoding="utf-8") as file_out:
                for sample in tqdm(res, desc="Output"):
                    if len(sample["relation_list"]) == 0:
                        continue
                    json_line = json.dumps(sample, ensure_ascii=False)
                    file_out.write("{}\n".format(json_line))

    # score
    if test_config["score"]:
        score_dict = {}
        correct = hyper_parameters_test["match_pattern"]
        #     correct = "whole_text"
        for file_path, pred_samples in res_dict.items():
            run_id = file_path.split("/")[-2]
            file_name = re.search("(.*?)_res_\d+\.json", file_path.split("/")[-1]).group(1)
            gold_test_data = ori_test_data_dict[file_name]
            prf = get_test_prf(pred_samples, gold_test_data, pattern=correct)
            if run_id not in score_dict:
                score_dict[run_id] = {}
            score_dict[run_id][file_name] = prf
        print("---------------- Results -----------------------")
        pprint(score_dict)


