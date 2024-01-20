# Codul licentei

```bash
alias py="python3"
# install deps

brew install pyenv
py -m pip install virtualenv       


# nonnecesar daca ai folderul licenta, dar am creat un pyenv cu:
# py -m venv licenta    

# foloseste virtualenv-ul creat,
source licenta/bin/activate    

mkdir -p config source test utils && touch main.py requirements.txt config/__init__.py source/__init__.py test/__init__.py utils/__init__.py


```