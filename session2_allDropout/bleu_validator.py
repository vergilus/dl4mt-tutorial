import os 
import sys 
import subprocess

class BleuValidator:
    def __init__(self, options, **kwords):
        ''' initiate a bash command for BLEU validation:
        first change work directory and then translate the validation set.
        for translte: 
        THEANO_FLAGS=device=gpu0 python -u  translate_gpu.py -n model src_dic trg_dict src saveto
        
        for multi-bleu.perl: perl multi-bleu.perl  MT02\en < MT02.trans
        '''
        # command for validation set translation 
        self.src_dict = options['dictionaries'][0]
        self.trg_dict = options['dictionaries'][1]
        self.valid_src = options['valid_datasets'][0]
        self.valid_trg = options['valid_datasets'][1]
        self.valid_freq = options['validFreq']*10
        
        self.valid_path=kwords['valid_path']
        self.temp_dir = kwords['temp_dir']
        self.translate_script = kwords['translate_script']
        self.bleu_script = kwords['bleu_script']
        
        # create a temp_dir (-p indicates parent directory if needed)
        os.system('mkdir -p %s' % self.temp_dir)
        self.check_script() # check BLEU script and proceed to BLEU validation
    
    def build_bleu_cmd(self, trans_file):
        return "perl %s %s < %s" \
                %(self.valid_path+self.bleu_script,
                  self.valid_trg,
                  trans_file
                  )
        
    def decode(self, device, trans_saveto, model_file):
        ''' 
        :cmd python translate_gpu.py -n model src_dic trg_dict src saveto
        :return a new process decoding validation file
        translate.py is usually in same directory
        '''
        cmd = "THEANO_FLAGS='device=%s' python %s -n -messageOff -k 2 %s %s %s %s %s" \
            %(device, 
              self.translate_script,
              model_file,
              self.src_dict,
              self.trg_dict,
              self.valid_src,
              os.path.join(self.temp_dir,trans_saveto)
              )
        print 'running: %s' %cmd
        return subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True)
    @staticmethod
    def parse_bleu_result(bleu_result):
        '''
        :input ['BLEU = 22.57, 58.2/30.1/17.5/10.5 (BP=0.947, ratio=0.949, hyp_len=26795, ref_len=28245)']
        :return float(22.57) or -1 for error
        '''
        print bleu_result
        bleu_result=bleu_result[0].strip()
        if bleu_result =='':
            return -1.
        try:
            bleu = bleu_result[:bleu_result.index(',')].split()[2]
            bleu = float(bleu)
        except ValueError:
            bleu=-1
        return bleu
    
    def testBLEU(self, trans_file):
        '''
        return BLEU for the sample validation
        :cmd = perl multi-bleu.perl  MT02\en < MT02.trans
        :return float(33.55) or -1 for error
        bleu is usually in the ../data/validate/
        '''
        cmd = self.build_bleu_cmd(self.temp_dir + trans_file)
        popen = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True)
        popen.wait()
        bleu=BleuValidator.parse_bleu_result(popen.stdout.readlines())
        if bleu==-1:
            print 'Fail to run script:%s for file:%s'% (self.bleu_script,trans_file)
            sys.exit(0)
        return bleu
    def remove_temp_file(self, model_file,trans_file):
        '''
        clear the temporal model file used in BLEU validation 
        :input = uidx 
        :command = rm temp_dir+trans_saveto  model_file
        '''
        cmd = 'rm %s  %s'%(self.temp_dir+trans_file,
                          model_file)
        #print cmd
        os.system(cmd)
        return   
    def check_script(self): 
        # check the BLEU script and the path
        if not os.path.exists(self.valid_path+self.bleu_script):
            print 'bleu script not exists: %s' % self.bleu_script
            sys.exit(0)
        if not os.path.exists(self.valid_src):
            print 'valid src file not exists: %s' % self.valid_src
            sys.exit(0)
        # check validation answers in the given path
        if os.path.exists(self.valid_trg):
            cmd = self.build_bleu_cmd(self.valid_trg)
        elif os.path.exists(self.valid_trg + str(0)):
            cmd = self.build_bleu_cmd(self.valid_trg + str(0))
        else:
            print 'valid trg file not exists: %s or %s0' % (self.valid_trg, self.valid_trg)
            sys.exit(0)
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

        popen.wait()
        bleu = BleuValidator.parse_bleu_result(popen.stdout.readlines())
        if bleu == -1:
            print 'Fail to run script: %s. Please CHECK' % self.bleu_script
            sys.exit(0)
        print 'Successfully test bleu script: %s' % self.bleu_script
    
    def check_save_decode(self,uidx):
        '''
        check whether to save or decode
        :input udix
        :return save or not  // decode or not 
        '''
        save = False
        decode = False
        if uidx % self.save_freq ==0:
            save = True
        if udix % self.valid_freq ==0:
            decode=True
        
        return save, decode
        