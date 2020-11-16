import requests
import logging
import base64
import time
import random
import binascii
from src.naivebayesmodel import *
from src.linsvmmodel import *
from src.mlpmodel import *

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')

                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', None)
        self.ans  = r.get('target', 'unknown')
        return r


if __name__ == "__main__":
    # create the server object
    s = Server()

    #mymodel = train_naive_bayes("data/datafile100.json", grams=[3], tf_idf=False)
    mymodel = train_SVM("data/datafile150.json", grams=[2,3], tf_idf=True)
    #mymodel = train_MLP("data/datafile150.json", grams=[3], tf_idf=True)
    for _ in range(2000):
        # query the /challenge endpoint
        s.get()
        myinp = binascii.hexlify(s.binary)
        target = mymodel.predict(str(myinp), s.targets)
        s.post(target)

        s.log.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]".format(target, s.ans, s.wins))
        # 500 consecutive correct answers are required to win
        if s.hash:
            with open('hashes.txt', 'w+') as f:
                f.write(s.hash)
            s.log.info("You win! {}".format(s.hash))

    s.session.close()