def annotations = getAnnotationObjects()
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
//println gson.toJson(annotations)

// Get the imageData & server
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

String path = server.getPath()

// automatic output filename, otherwise set explicitly
rt_path = "/Users/lucius_mac/Desktop/hubmap-kidney-segmentation/"
//fname = "test_image"
//println path[path.lastIndexOf(':')+1..-5]

//outfname = rt_path+path[path.lastIndexOf(':')+93..-5]+".json"
outfname = rt_path+"test_1.json"
//println outfname

File file = new File(outfname)
file.withWriter('UTF-8') {
    gson.toJson(annotations,it)
}