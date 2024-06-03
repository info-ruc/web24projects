import os
import re
import numpy
from collections import *
import pandas as pd
import binascii

def getOpcodeSequence(filename):
    opcode_seq = []
    p = re.compile(r'\s([a-fA-F0-9]{2}\s)+\s*([a-z]+)')
    with open(filename) as f:
        for line in f:
            if line.startswith(".text"):
                m = re.findall(p,line)
                if m:
                    opc = m[0][1]
                    if opc != "align":
                        opcode_seq.append(opc)
    return opcode_seq

def getOpcodeNgram(ops, n=3):
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram

def train_opcode_lm(ops, order=4):
    lm = defaultdict(Counter)
    prefix = ["~"] * order
    prefix.extend(ops)
    data = prefix
    for i in xrange(len(data)-order):
        history, char = tuple(data[i:i+order]), data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.iteritems()]
    outlm = {hist:chars for hist, chars in lm.iteritems()}
    return outlm


def getImagefrom_bin(filename, width = 512, oneRow = False):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    if oneRow is False:
        rn = len(fh)/width
        fh = numpy.reshape(fh[:rn*width],(-1,width))
    fh = numpy.uint8(fh)
    return fh

def getMatrixfrom_asm(filename, startindex = 0, pixnum = 5000):
    with open(filename, 'rb') as f:
        f.seek(startindex, 0)
        content = f.read(pixnum)
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    fh = numpy.uint8(fh)
    return fh

def getMatrixfrom_hex(filename, width):
    hexar = []
    with open(filename,'rb') as f:
        for line in f:
            hexar.extend(int(el,16) for el in line.split()[1:] if el != "??")
    rn = len(hexar)/width
    fh = numpy.reshape(hexar[:rn*width],(-1,width))
    fh = numpy.uint8(fh)
    return fh

def read_hexbytes(filename):
    hexar = []
    with open(filename,'rb') as f:
        for line in f:
            hexar.extend(int(el,16) for el in line.split()[1:] if el != "??")
    rn = len(hexar)/256
    fh = numpy.reshape(hexar[:rn*256],(-1,256))
    fh = numpy.uint8(fh)
    return fh

def asm2img(basepath) :
	mapimg = defaultdict(list)
	subtrain = pd.read_csv('subtrainLabels.csv')
	i = 0
	for sid in subtrain.Id:
		i += 1
		print "dealing with {0}th file...".format(str(i))
		filename = basepath + sid + ".asm"
		im = getMatrixfrom_asm(filename, startindex = 0, pixnum = 1500)
		mapimg[sid] = im

	dataframelist = []
	for sid,imf in mapimg.iteritems():
		standard = {}
		standard["Id"] = sid
		for index,value in enumerate(imf):
			colName = "pix{0}".format(str(index))
			standard[colName] = value
		dataframelist.append(standard)

	df = pd.DataFrame(dataframelist)
	df.to_csv("imgfeature.csv",index=False)

def asm2ngrams(basepath) :
	map3gram = defaultdict(Counter)
	subtrain = pd.read_csv('subtrainLabels.csv')
	count = 1
	for sid in subtrain.Id:
		print "counting the 3-gram of the {0} file...".format(str(count))
		count += 1
		filename = basepath + sid + ".asm"
		ops = getOpcodeSequence(filename)
		op3gram = getOpcodeNgram(ops)
		map3gram[sid] = op3gram

	cc = Counter([])
	for d in map3gram.values():
		cc += d
	selectedfeatures = {}
	tc = 0
	for k,v in cc.iteritems():
		if v >= 500:
			selectedfeatures[k] = v
			print k,v
			tc += 1
	dataframelist = []
	for fid,op3gram in map3gram.iteritems():
		standard = {}
		standard["Id"] = fid
		for feature in selectedfeatures:
			if feature in op3gram:
				standard[feature] = op3gram[feature]
			else:
				standard[feature] = 0
		dataframelist.append(standard)
	df = pd.DataFrame(dataframelist)
	df.to_csv("3gramfeature.csv",index=False)

if __name__ == "__main__" :
	basepath = "/media/user/VM/VMForce_TMP/web/data/subtrain/"
	asm2img(basepath)
	asm2ngrams(basepath)
	
