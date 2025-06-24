from mlforkidsnumbers import MLforKidsNumbers

project = MLforKidsNumbers(
    key="04f52f80-460a-11f0-bd2f-b5647c7927efb82e43b7-5d44-43cc-91ff-21653b77ab48",
    modelurl="https://mlforkids-newnumbers.j8clybxvjr0.us-south.codeengine.appdomain.cloud/saved-models/8e718660-4609-11f0-bd2f-b5647c7927ef/status"
)

# CHANGE THIS to something you want your
# machine learning model to classify
testvalue = {
    "Pattern" : 0,
}

response = project.classify(testvalue)
top_match = response[0]

label = top_match["class_name"]
confidence = top_match["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))
