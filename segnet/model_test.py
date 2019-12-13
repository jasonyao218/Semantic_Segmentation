from segnet import segnet

model = segnet(2,input_height=416,input_width=416)
model.summary()
