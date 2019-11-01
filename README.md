# usage

Install requirements with virtualenv.

First run `load.py` normalize and add bias to inputs. It then reads inputs and targets into binary files. Targets get processed into arrays with a '1' indicating target.

Run `train.py`. Takes 2 arguments: learning rate and epochs.

Both files assume `/MNIST/` in repo directory.

## `virtual env`

    pip3 install virtualenv

    virtualenv -p python3 env

This will install virtualenv using pip3 and create a an `env` folder that will contain the libraries it uses.

    source env/bin/activate

This will put your shell into the virtual environment where you will have the libraries brought it. *(Note: fish* `. env/bin/activate.fish*)*

    pip3 install -r requirements.txt

This will install the libraries that are found in `requirements.txt`
When you are done working, use `deactivate` to get out of the virtual environment

### log

~~sort arrays. (ie is it x[i][j] or other way around?)~~

#### 10/24
~~Can weight updates be done with matrix ops?~~ *YES*
  ~~Test arrays with Python3 interpreter~~
~~Working out algorithm~~
