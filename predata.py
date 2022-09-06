# %%

dirr = 'FNN'
gosfk = 'politifact_fake'
dataname = os.path.join(dirr, gosfk)
gosrl=   'politifact_real'
datanamerl = os.path.join(dirr, gosrl)
post = []
replyy = []
rumor =[]
for f in os.listdir(dataname):
    fn = os.path.join(dataname, f)
    for f1 in os.listdir(fn):
        if(f1 == 'news_article.json'):
            fileop = os.path.join(dataname, f,f1)
            fileop = open(fileop)
            dataj = json.load(fileop)
            prep_post = preprocess.preprocess_text(str(dataj.get('text')))
            post.append(prep_post) 
        if(f1 == 'replies.json'):
            fileop = os.path.join(dataname, f,f1)
            fileop = open(fileop)
            dataj = json.load(fileop)
            temp = []
            for key,value in dataj.items():
                if type(value) == type(dict()):
                    for key1,value1 in value.items():
                        prep_post = preprocess.preprocess_text(value1['text'])
                        temp.append(prep_post)
            replyy.append(temp)
    rumor.append(0)  
    
                    
for f in os.listdir(datanamerl):
    fn = os.path.join(datanamerl, f)
    for f1 in os.listdir(fn):
        if(f1 == 'news_article.json'):
            fileop = os.path.join(datanamerl, f,f1)
            fileop = open(fileop)
            dataj = json.load(fileop)
            prep_post = preprocess.preprocess_text(str(dataj.get('text')))
            post.append(prep_post) 
        if(f1 == 'replies.json'):
            fileop = os.path.join(datanamerl, f,f1)
            fileop = open(fileop)
            dataj = json.load(fileop)
            temp = []
            for reply in dataj:
                   for tx in dataj[reply]:
                        val = str(tx.get('text'))
                        if val != None:
                            prep_post = preprocess.preprocess_text(val)
                            temp.append(prep_post)
                        else:
                            temp.append([])
                        
                            
                        
            replyy.append(temp)
    rumor.append(1)         
savehere =  os.path.join(dirr,'prep_pl','news.txt' )
with open(savehere, 'w+',encoding="utf-8") as f:
    for tex in post:
        f.write(tex)
        f.write('\n')

rep_file_name = 'reply'
rep_dir = 'reply'
count = 0
for f in replyy:
    rep_file_name1 = rep_file_name + str(count)
    rep_file_name1 += '.txt' 
    fileop = os.path.join(dirr,'prep_pl',rep_dir, rep_file_name1)
    
    with open(fileop, 'w+',encoding="utf-8") as ff:
        for tex in f:
            ff.write(tex)
            ff.write('\n')
    count +=1
