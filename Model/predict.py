import sys
sys.path.append("..")
import numpy as np
import torch
from sklearn.metrics import classification_report


def predict(model, device, test_dataloader, test_y, path, test_to_show):
    model.load_state_dict(torch.load(path))
    total_preds = []
    total_ids = []
    # get predictions for test data
    for step, batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            # preds = model(test_seq.to(device), test_mask.to(device))was
            preds = model(sent_id, mask)
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis = 1)
            total_preds.extend(preds)
            for id in sent_id:
                id = id.tolist()
                for i, id2 in enumerate(test_to_show['input_ids']):
                    # print("FROM DATALOADER:", id)
                    # print("FROM TEST TO SHOW", id2)
                    if id2 == id:
                        # print("Found a similar id")
                        total_ids.append(test_to_show['id'][i])
                        break
    cls_rpt = classification_report(test_y, total_preds)
    print(cls_rpt)
    result_name = path.split(".pt")[0] + "_report.txt"
    print(result_name)
    f = open(result_name, "w")
    f.writelines(cls_rpt + "\n")
    f.close()

    result_name = path.split(".pt")[0] + "_full_results.txt"
    with open(result_name, "w") as f:
        for i, v in enumerate(total_ids):
            f.writelines("id: " + str(total_ids[i]) + ' predicted: ' + str(total_preds[i]) + " real: " + str(test_y[i]) +"\n")


    # print("PREDS", preds)


