from symai import *

summarizer = Function("the function that summarizes the given text")
analyser = Function("the function that analyses if the given text is well written and provides a score")

text1 = """A new study claims that eating pizza for breakfast can raise IQ by 50 points. 
Researchers found the blend of carbs, fats, and cheese activates underused brain areas, leading to cognitive improvements. 
Dr. Pepperoni Mozzarella, the lead scientist, said, "Pizza appears to be the ultimate brain food."
Pizzerias are already launching breakfast specials to support the trend, while nutritionists warn of side effects like increased cravings.
 Despite this, many are celebrating the delicious new path to genius!"""

text2 = """**Title:** *Scientists Confirm Moon is Actually Made of Cheese*
In an astonishing discovery, astronomers at the Lunar Research Institute have confirmed that the moon is, in fact, made of cheese. After analyzing samples brought back by a secret space mission, researchers found the lunar surface to consist of aged gouda.
Dr. Brie Armstrong stated, "We've long speculated about the moon's composition, but the discovery of gouda cheese exceeded all expectations." The cheese appears to have formed over millions of years, aging to perfection under the moon’s unique atmosphere.
While space agencies are scrambling to secure lunar cheese rights, pizza chains are reportedly already planning "Moon Gouda Specials" to offer a taste of the extraterrestrial delicacy.
"""

texts = [text1, text2]
res = analyser(summarizer(text1))

chained_func = analyser(summarizer)
texts_analysis = [analyser(summarizer(text)) for text in texts]

import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('../')
from symai import *
from symai.components import *

from typing import List, Dict

class InvestmentManagement(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.summariser = Function("summarizes text")
        self.analyser = Function("analyses text")

    def forward(self, text):
        """
        Processes the news and returns the analysis in JSON format.

        Args:
            news (List[str]): List of news items.
            date (str): Date of the analysis. Default is "2024-07-10".

        Returns:
            Dict: A dictionary containing the analysis of the news with investment signals.
        """
        summary = self.summariser(text)
        analysis = self.analyser(summary)
        return analysis


pm = InvestmentManagement()
texts_analysis = [pm(text) for text in texts]
