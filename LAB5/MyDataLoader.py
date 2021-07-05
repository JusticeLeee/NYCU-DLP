import torch


class MyDataLoader:
    def __init__ (self, path, train):
        
        #build dictionary
        self.char2idx = self.build("c2i") 
        self.idx2char = self.build("i2c")
        self.tense2idx= {'sp':0, 'tp':1, 'pg':2, 'p':3 }
        self.idx2tense= { 0:'sp', 1:'tp', 2:'pg', 3:'p'} 
        self.train = train    

        self.words, self.tenses = self.get_dataset(path,self.train)
        
        assert len(self.words)==len(self.tenses),'word list is not compatible with tense list'
        
    def build(self,type):
        
        if(type=="c2i"):
            #first two 
            dictionary={'SOS':0,'EOS':1}
            #alhpabets
            dictionary.update([(chr(i+97),i+2) for i in range(0,26)])
            return dictionary
        if(type=="i2c"):
            #first two
            dictionary={0:'SOS',1:'EOS'}
            #alphabets
            dictionary.update([(i+2,chr(i+97)) for i in range(0,26)])
        return dictionary
    
    def string2tensor(self,string,EOS=True):
        
        indices=[self.char2idx[char] for char in string]
        if EOS:
            indices.append(self.char2idx['EOS'])
        return torch.tensor(indices,dtype=torch.long).view(-1,1)

    def tense2tensor(self,tense):
        return torch.tensor([tense],dtype=torch.long)

    def tensor2string(self,tensor):
        string=""
        string_length=tensor.size(0)
        for i in range(string_length):
            char=self.idx2char[tensor[i].item()]
            if char=='EOS':
                break
            string+=char
        return string
    
    def __len__(self):
        return len(self.words)
    
    def get_dataset(self,path,train):
        words=[]
        tenses=[]
        with open(path,'r') as file:
            if train:
                #train.txt
                for line in file:
                    words.extend(line.split('\n')[0].split(' '))
                    tenses.extend(range(0,4))
            else:
                #test.txt
                for line in file:
                    words.append(line.split('\n')[0].split(' '))
                test_tenses=[['sp','p'],['sp','pg'],['sp','tp'],['sp','tp'],['p','tp'],['sp','pg'],['p','sp'],['pg','sp'],['pg','p'],['pg','tp']]
                for test_tense in test_tenses:
                    tenses.append([self.tense2idx[tense] for tense in test_tense])
        return words,tenses
    
    def __getitem__(self, idx):

        if self.train:
            return self.string2tensor(self.words[idx],EOS=True),self.tense2tensor(self.tenses[idx])
        else:
            return self.string2tensor(self.words[idx][0],EOS=True),self.tense2tensor(self.tenses[idx][0]),\
                   self.string2tensor(self.words[idx][1],EOS=True),self.tense2tensor(self.tenses[idx][1])
        
