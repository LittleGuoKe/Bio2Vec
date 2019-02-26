import sentencepiece as spm


class segment:
    def __init__(self):
        self.model_path = '/home/yangshan/PycharmProject/home/yangshan/PycharmProject/Bioinformatics/protein2vec/new_data/sentencePieceModel20000bpe.model'

    def train_model(self):
        spm.SentencePieceTrainer.Train(
            '--input=/home/yangshan/PycharmProject/home/yangshan/PycharmProject/Bioinformatics/protein2vec/new_data/merge_protein.txt --model_prefix=/home/yangshan/PycharmProject/home/yangshan/PycharmProject/Bioinformatics/protein2vec/new_data/sentencePieceModel20000unigram --vocab_size=20000 --model_type=unigram')

    def segment_protein(self, protein):
        sp = spm.SentencePieceProcessor()
        sp.Load(self.model_path)
        print(sp.EncodeAsPieces(protein))


'''Test model'''
# protein = "MFDHDVEYLITALSSETRIQYDQRLLDEIAANVVYYVPRVKSPDTLYRL"
# segment = segment()
# segment.segment_protein(protein)
# ['▁MFD', 'HDVEY', 'LITA', 'LSS', 'ET', 'RI', 'QYD', 'QRLL', 'DEI', 'AANV', 'VYY', 'VP', 'RV', 'KSP', 'DTLY', 'RL']
'''Train model'''
segment = segment()
segment.train_model()
