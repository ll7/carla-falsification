## Mistakes that can be avoided
Some Mistakes can very time consuming although it can be avoided very easily
### Versions:
For RL it is common to use different Frameworks like pyTorch, Keras and others. Some of the Frameworks have dependencies to other Software versions therefore it is really important that you don't just install the latest version of everything and hope that everything is compatible. 
It is lot more work to properly remove a version rather than directly installing the right one. 
In my case I had Problems with dependencies between Python, Cuda, pyTorch, and compatible drives for Cara. 

For my Reinforcement Learning setup, I use Ubuntu 20.04 Python 3.8.10, Cuda 11.6, NVidia Driver Version: 510.73.05 and for PytTorch the Preview(Nightly) build for Cuda 11.3.
The Computer that I use (imech031) have a RTX 3800 Ti and an AMD Ryzen 9 5900X 12-Core Processor

Important: If you are stuck in a version problem, don't be afraid of asking another person or your supervisor, sometimes it is easier to solve a problem together or it came up before and can solved really fast. 


### Test Coverage 
Like many others I usually have no desire to write Test Scenarios and test the code properly at the beginning of a project. In many cases it worked without huge test corvarage but if you have a bug and can't find it, it is often much more time consuming then writing the tests in the first place. 
My suggestion is to write some basic test coverage and think from time to time what you want to reach and witch tests could support to do so. 
In my case an early test for the determinism of the walker would have me saved many hours of debugging. 

### Use of variables

