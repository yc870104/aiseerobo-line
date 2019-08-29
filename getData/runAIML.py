# import aiml
from getData.aiml.Kernel import Kernel  # Use PyAiml dev
import os
import sys


# Create a Kernel object.
def runAIML(inpt):
    kernel = Kernel()
    kernel.bootstrap(brainFile=os.path.dirname(__file__)+"/aiml-doc/aiwisfin_brain.brn")
    # kernel.bootstrap(brainFile="aiml-doc/aiwisfin_brain.brn")
    return kernel.respond(inpt)


print(runAIML('我愛你'))
