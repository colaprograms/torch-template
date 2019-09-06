from nets.ctcnet import CTCModel
from util.beam import beamsearcher
from util.file import FileHolder, labeled_file, file_list_to_tensor
import matplotlib.pyplot as p
import torch
import numpy as np

CHECKPOINTS = [
    "checkpoint-0049-0.49-CTCModel",
    "checkpoint-0046-0.43-CTCModel", "checkpoint-0079-0.42-CTCModel"
]

def show(inp):
    #print(outp[0])
    p.imshow((inp * 0.2 + 0.9).squeeze(0).permute(1, 2, 0))
    p.show()

class Test:
    def __init__(self, file, quiet=False):
        self.model = CTCModel().cuda()
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model'])
        # set to evaluation mode!!!
        self.model.eval()
        self.fh = FileHolder()
        self.quiet = quiet

    def test(self):
        count = 0
        nimages = self.fh.nvalidation()
        print("Testing", nimages, "images")
        BATCHSIZE = 64
        for j in range(0, nimages, BATCHSIZE):
            paths = []
            outputs = []
            for filename, val in self.fh.info['validation'][j:j+BATCHSIZE]:
                path = labeled_file(filename)
                paths += [path]
                outputs.append(val)
            inputs = file_list_to_tensor(paths)

            logits = self.model(inputs)
            logits = logits.detach().cpu().numpy()
            for i in range(len(outputs)):
                answer = beamsearcher(logits[i, :, :], 8)
                if answer[0].str() != outputs[i]:
                    if not self.quiet:
                        print("%s image was wrong!" % nth(j + i))
                        print("answer", answer[0].str())
                        print("correct:", outputs[i])
                        print("top 3 guesses:", [z.str() for z in answer[:3]])
                        print("image:")
                        show(inputs[i, :, :, :])
                else:
                    count+=1
            i = j + len(outputs)
            print("Tested %3d/%d. %.2f%% correct" % (i, nimages, count/i*100))
        return count, nimages

    def classify_files(self, paths):
        inputs = file_list_to_tensor(paths)
        logits = self.model(inputs)
        logits = logits.detach().cpu().numpy()
        return inputs, beamsearcher(logits, 8)

    def decode_one_image(self, filename):
        inputs, answer = self.classify_files([filename])
        print("Answer:", answer[0].str())
        print("Top 3 guesses:", [str(z) for z in answer[:3]])
        print("Image:")
        show(inputs[0, :, :, :])

SUFFIX = ["th", "st", "nd", "rd"] + ["th"] * 6
def nth(j):
    "Return j with an appropriate ordinal suffix. e.g. nth(3) == '3rd'"
    suffix = SUFFIX[j % 10]
    if 11 <= j < 19:
        suffix = "th"
    return "%d%s" % (j, suffix)

def test(checkpoint, quiet=False):
    count, m = Test(checkpoint, quiet).test()
    return "Total correct: %.2f%% (%d/%d)" % (count / m * 100, count, m)
