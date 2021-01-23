l1 = ['112 1235g gdfg dfg45646 dfgdf5.', '123 4658df df454646 45fd4d5', '22222 22222 22222']
stop_word = ['112', 'gdfg']
print(len(l1))
print(len(l1[1:]))

atrtemp=l1[1:]
if len(atrtemp) !=0:
    atrtemp = ' '.join(atrtemp)
    print(atrtemp)

for i in stop_word:
    stop0 = i + " " 
    stop1 = " "+ i + " "
    stop2 = " " + i + "."
    stop3 = " " + i
    l1[0] = l1[0].replace(stop0,'')
    l1[0] = l1[0].replace(stop1,' ')
    l1[0] = l1[0].replace(stop2,'')
    l1[0] = l1[0].replace(stop3,'')
print(l1)