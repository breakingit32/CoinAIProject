class Config():
    def __init__(self):
        pass
    
    num_classes=3
    labels_to_class = {0:'0',1:'180',2:'276'}
    class_to_labels = {'0':0,'180':1,'276':2}
    resize = 28	
    num_epochs =1
    batch_size =128