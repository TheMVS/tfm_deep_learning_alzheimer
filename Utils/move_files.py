import os, shutil

WD = '/Users/Puente/Desktop/'
destination = 'test'
origins = ['disc6','disc7','disc8','disc9','disc10','disc11','disc12']

for folder in origins:
	files = os.listdir(WD + '/' + folder)
	for f in files:
		src = WD+'/'+folder+'/'+f+'/RAW/'+f+'_mpr-1_anon_sag_66.gif'
		dst = WD+'/'+destination+'/'
		shutil.move(src,dst)
