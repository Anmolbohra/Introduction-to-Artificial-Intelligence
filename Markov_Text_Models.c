# -*- coding: utf-8 -*-
"""
Intro to AI -- Analyzing and Generating Text
Nils Napp
University at Buffalo 
"""

from collections import defaultdict,deque,Counter
import numpy as np
import random
import pickle
import re
import random

class Ngrams:
    
    def __init__(self,size=9):
        
        assert size>0
        self.maxGram=size-1

        #make training counters
        #could use defaultdict, but then lookups create 0 during testing
        self.ngrams=[]
        for i in range(size):
            self.ngrams.append(Counter())
    
        #call function to set up the character maps
        self._setup() 
        
        #store list of files used during training
        self.fnames=[]
        
    '''
    Read a file and add it to the ngrams 
    '''
    def slurpFile(self, fname : str):
        #Go over file a number o times
        
        if not fname in self.fnames:
        
            self.fnames.append(fname)
            
            for gramSize in range(self.maxGram+1):
                
                '''
                dirty char iter ---> Filter (map to charset) ---> ngram iter
                '''
                print(str(gramSize-1) + '-grams: ', end='')
                
                dirtyCharacterIter=self.charIter(fname)
                cleanCharacterIter=self.cleanIter(dirtyCharacterIter)
                
                giter=self.gramIter(cleanCharacterIter, n=gramSize+1)
                
                self.ngrams[gramSize].update(giter)
                
                print(str(sum(self.ngrams[gramSize].values())))

        else:
            print("File already contained in ngram data: " + fname)
                
    def _setup(self):
        
        #These strings are used to construct a character dict
        #that makes the clean character iterator
        
        abc='abcdefghijklmnopqrstuvwxyz'
        ABC=abc.upper()
        white=' \n\t\r\f\v_'
        punct=".,!?'()"
        quotes='"`“”'  # all the other things that should show up as '
        
        keys=abc+ABC+white+punct+quotes
        vals=abc+abc+'       '+punct+"''''"

        self.charDict=defaultdict(lambda: '',list(zip(iter(keys),iter(vals)))) 

        #remove duplicates
        #these are the characters that can 
        #show up in the clean char iterator
        self.chars=tuple(set(vals))


    def saveGrams(self,fname):
        '''
        grams    : NGRAMS
        chars    : CHARS
        charDict : CHAR_DICT
        maxGram  : MAX_GRAMS
        fileNames: FILE_NAMES
        '''
        
        saveDict=dict([  ['NGRAMS',self.ngrams]
                        ,['CHARS',self.chars]
                        ,['CHAR_DICT',dict(self.charDict)]
                        ,['MAX_GRAMS',self.maxGram]
                        ,['FILE_NAMES',self.fnames]])
        
        with open(fname,'wb') as fh:
            pickle.dump(saveDict,fh)
        

    def loadGrams(self,fname):

        with open(fname,'rb') as fh:
            loadDict=pickle.load(fh,encoding='bytes')            
        
        self.ngrams=loadDict['NGRAMS']
        self.chars=loadDict['CHARS']
        self.charDict=defaultdict(lambda: '', loadDict['CHAR_DICT'])
        self.maxGram=loadDict['MAX_GRAMS']
        self.fnames=['FILE_NAMES']

            
    '''
    return a n iterator that returns single charactres from a file
    '''
    def charIter(self, fname : str):
        
        with open(fname,'r',encoding='UTF8') as fh:
        
            c=fh.read(1)
            
            while not c == '':
                yield c
                c=fh.read(1)                

    '''
    iterator that returns a cleaned single character
    from an interator that returns non-cleand characters 
    '''
    def cleanIter(self, singleCharIter : iter):
           
        cnext=next(singleCharIter)
        
        skipWhite=False
        
        while not cnext == '':
            
            if skipWhite:
                if self.charDict[cnext]=='' or self.charDict[cnext]==' ':
                    #print('.')
                    pass
                else:
                    skipWhite=False
                    yield self.charDict[cnext]
            else:
                if self.charDict[cnext]==' ':
                    skipWhite=True
                    #print('Start Skipping')
                    yield ' '
                elif self.charDict[cnext]=='':
                    pass
                else:
                    yield self.charDict[cnext]
            
            cnext=next(singleCharIter)

                       
    '''
    iterator that returns cleaned n-grams
    '''
    def gramIter(self, charIter, n : int = 1):
        
        #single character iterator that cleans inputs:
        #Only lower case cahracters
        #All white spaces mapped to single ' ' 
        #Limited set of punctuation marks <--- fix to include '
                
        #two sided que to shift in and out characters to make n-grams 
        ngramQue=deque()
        
        
        for i in range(n-1):
            c=next(charIter)
            ngramQue.append(c)                                    
    
        for c in charIter:
            ngramQue.append(c)
            ngram=''.join(e for e in ngramQue)
            yield ngram
            ngramQue.popleft()
   

    def probLastChar(self, gram : str) -> float:
    
            prefix=gram[:-1]
            prefixLen=len(prefix)
    
            total=0
            
            count = self.ngrams[prefixLen][gram]
            
            for c in self.chars:            
                total = total + self.ngrams[prefixLen][prefix+c]
    
            if count == 0:
                return self.probLastChar(gram[1:])
            else:
                return 1.0*count/total
            
    def logProbSeq(self, text, prevLen : int = 1):

         prob=0
         gi = self.gramIter(self.cleanIter(iter(text)),prevLen+1)

         for gram in gi:
             prob= prob + np.log2(self.probLastChar(gram))

         return prob
         
    '''
     Should be low if ngram model and the actual text agree
    '''
    def avgEntropy(self, text, prevLen : int =1):

        prob=0
        ccount=0
        gi = self.gramIter(self.cleanIter(iter(text)),prevLen+1)

        for gram in gi:
            ccount=ccount+1
            prob= prob + np.log2(self.probLastChar(gram))

        return -prob/ccount


    def substitute(self, orig : str, subs :str, text :str) -> str:
        outText = ''
        
        assert len(orig) == len(subs)
        
        subsDict=dict(zip(orig,subs))
    
        for c in self.cleanIter(iter(text)):
            
            if c in orig:
                outText = outText + subsDict[c]
            else:
                outText = outText + c
        return outText        

    '''
    Return a random character 
    following the given gram according to the n-grams statistics
    
    If the gram + c was never observed for any c then try return 
    a random sample from the last n-1 characters of the n-gram  
    '''
    def nextChar(self, gram : str)->str:
        pass


    def grow(self, text : str, gram : int = 4)->str:
        pass
    
    '''
    You don't need to use the above helper functions
    Only makeSentence will be evaluated.
    '''

    def getNGramSentenceRandom(gram, word, n=50):
        for i in range(n):
            print(word, )
            choices = [element for element in gram if element[0][0] == word]
            if not choices:
                break
            word = weighted_choice(choices)[1]

    def weighted_choice(choices):
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w > r:
                return c
            upto += w

    def generateNgram(self, n):
        gram = dict()
        assert n > 0 and n < 100

        for i in range(len(words) - (n - 1)):
            key = tuple(words[i:i + n])
            if gram.has_key(key):
                gram[key] += 1
            else:
                gram[key] = 1

        #        gram = sorted(gram.items(), key=lambda (_, count): -count)
        return gram
    def makeSentence(self, gram : int = 4)->str:
        with open("originspecies.txt") as f:
            txt = f.read()
        words = re.split('[^A-Za-z]+', txt.lower())
        words = filter(None, words)
        gram1 = set(words)
        gram1_iter = iter(gram1)
        trigram = self.generateNgram(3)
        print (trigram[:20])
        print ("Generating %d-gram list..." % 5,)
        ngram = self.generateNgram(5)
        print ("Done")

        for word in ['and', 'he', 'she', 'when', 'john', 'never', 'i', 'how']:
            print (" %d-gram: \"" % 5)
            getNGramSentenceRandom(ngram, word, 15)
            print ("\"")

        print ("Generating %d-gram list..." % 5,)
        gram10 = generateNgram(10)
        print ("Done")

        for word in ['and', 'he', 'she', 'when', 'john', 'never', 'i', 'how']:
            print(" %d-gram: \"" % 5,)
            getNGramSentenceRandom(ngram, word, 100)
            print("\"")



    def decode(text : str):
        return 'orig','subs'

if __name__ == '__main__':
        
    ng=Ngrams()
    
    #load the grams from a stored file
    #you can also add some new text files with ng.slurpFile('some.txt')
    #or add any other text you find online 
    #For free books you can check out 
    #Project Gutenberg: https://www.gutenberg.org/
    #They have lots of books in plain text (sluprFile() assums UTF-8 encoding)
    ng.loadGrams('smallGrams.pkl')
    
    
    
    scrambledText  =  " xvkndui?d).xc,) nwwnxqv)x ?c,)wxvi" \
                 + ")xfjnw)nd)xd)?jwxdt?m)lwxbm)k?i,w) k,j" \
                 + ",)k,) xv)w,xtndu)x)ijxndndu)v,vvn?db)s" \
                 + "k,scndu)i nii,jm)qjb) nwwnxqvm)x)sy!,j" \
                 + "v,sojniy),hf,jim) xv)tnvqxy,t)i?)tnvs?" \
                 + "a,j)ikxi)k,)kxt)!,,d)ikjovi)ndi?)ik,)q" \
                 + "nttw,)?l)?d,)?l)ik,) ?jvi)v,sojniy)t,!" \
                 + "xsw,v),a,j)i?)!,lxww)xq,jnsxd)ndi,wwnu" \
                 + ",ds,b)qjb) nwwnxqv)kxt) jnii,d)?d)knv)" \
                 + "s?qfxdy)!w?u)x!?oi)ik,)vkxt? )!j?c,jvm" \
                 + ")x)qyvi,jn?ov)uj?of)ikxi)kxt)v?q,k? )?" \
                 + "!ixnd,t)qxdy)?l)ik,)kxscndu)i??wv)ik,)" \
                 + "odni,t)vixi,v)ov,t)i?)vfy)?d)?ik,j)s?o" \
                 + "dijn,vb)d? )ik,)uj?of)kxt)j,fwn,t)nd)x" \
                 + "d)xdujy)vsj,,t)?d)i nii,jb)ni)nt,dinln" \
                 + ",t)knq)s?jj,siwy)xv)x)l?jq,j)q,q!,j)?l" \
                 + ")ik,)dxin?dxw)v,sojniy)xu,dsyv)kxscndu" \
                 + ")uj?ofm)ixnw?j,t)xss,vv)?f,jxin?dvm)?j" \
                 + ")ibxb?bm)x).?!)k,)kxt)d?i)fo!wnswy)tnv" \
                 + "sw?v,tb)ik,d)ik,)vkxt? )!j?c,jv)xvi?dn" \
                 + "vk,t)knq)!y)tj?ffndu)i,skdnsxw)t,ixnwv" \
                 + ")ikxi)qxt,)sw,xj)ik,y)cd, )x!?oi)knukw" \
                 + "y)swxvvnln,t)kxscndu)?f,jxin?dv)ikxi)k" \
                 + ",)kxt)s?dtosi,tb)gik,y)kxt)?f,jxin?dxw" \
                 + ")ndvnuki)ikxi),a,d)q?vi)?l)qy)l,ww? )?" \
                 + "f,jxi?jv)xi)ibxb?b)tnt)d?i)kxa,mg)vxnt" \
                 + ")qjb) nwwnxqvm)d? ) nik)j,dtnin?d)ndl?" \
                 + "v,sm)x)sy!,jv,sojniy)lnjq)k,)l?odt,tb)" \
                 + "gn)l,wi)wnc,)nt)!,,d)cnsc,t)nd)ik,)uoi" \
                 + "b) k?,a,j) j?i,)iknv),nik,j) xv)x) ,ww" \
                 + "fwxs,t)ndvnt,j)?j)kxt)vi?w,d)x)w?i)?l)" \
                 + "?f,jxin?dxw)txixbg)ik,).?wi)i?)qjb) nw" \
                 + "wnxqv)lj?q)ik,)vkxt? )!j?c,jv)jnf?vi,)" \
                 + "xv)fxji)?l)x)qosk)!j?xt,j),xjik'oxc,)i" \
                 + "kxi)kxv)vkxc,d)ik,)dbvbxb)i?)niv)s?j,b" \
                 + ")sojj,di)xdt)l?jq,j)xu,dsy)?llnsnxwv)v" \
                 + "xy)ik,)vkxt? )!j?c,jv)tnvsw?voj,vm) kn" \
                 + "sk)!,uxd)nd)xouovi)m)kxa,)!,,d)sxixvij" \
                 + "?fkns)l?j)ik,)dbvbxbm)sxwwndu)ndi?)'o," \
                 + "vin?d)niv)x!nwniy)i?)fj?i,si)f?i,di)sy" \
                 + "!,j ,xf?dv)xdt)niv)a,jy)axwo,)i?)dxin?" \
                 + "dxw)v,sojniyb)ik,)xu,dsy)j,uxjt,t)xv)i" \
                 + "k,) ?jwtv)w,xt,j)nd)!j,xcndu)ndi?)xta," \
                 + "jvxjn,v)s?qfoi,j)d,i ?jcv)lxnw,t)i?)fj" \
                 + "?i,si)niv)? d"
    
    
    #Way to count the character sequences
    charCnt = Counter(scrambledText)

    #return a list of (char,cnt) pairs witht he most common
    #one as the first element
    charlist=charCnt.most_common()


    #If you need to gnerate bi-grams you can re-use the functions
    #used in reading files. For example the following text will go
    #through all the 4-grams in the scrambled text and pring the first 10
    pcnt=0
    for gram in ng.gramIter(iter(scrambledText),4):
        if pcnt < 10:
            print(gram)
        pcnt=pcnt+1
 
    
    #The iterators can be used directly fed into the Counter
    counts4gram=Counter(ng.gramIter(iter(scrambledText),4))
    
    #Display the 5 most common pairs
    print(counts4gram.most_common(5))
    
    #substitute
    print(ng.substitute('eol','leo','Hello there.' ))
    print(ng.nextChar("4"))