from datetime import datetime
from typing import Any, Callable, Optional

from ..components import Indexer
from ..memory import SlidingWindowStringConcatMemory
from ..symbol import Symbol


class CodeFormatter:
    def __call__(self, value: str, *args: Any, **kwds: Any) -> Any:
        # extract code from chat conversations or ```<language>\n{code}\n``` blocks
        return Symbol(value).extract('Only extract code without ``` block markers or chat conversations')


class Conversation(SlidingWindowStringConcatMemory):
    def __init__(self, init:     Optional[str] = None,
                 file_link:      Optional[str] = None,
                 index_name:     str           = Indexer.DEFAULT,
                 auto_print:     bool          = True,
                 token_ratio:    float         = 0.6, *args, **kwargs):
        super().__init__(token_ratio)
        self.token_ratio = token_ratio
        self.auto_print  = auto_print
        self.file_link   = file_link
        if init is not None:
            self.store_system_message(init, *args, **kwargs)
        if file_link is not None:
            self.store_file(file_link, *args, **kwargs)
        self.indexer = Indexer(index_name=index_name)
        self._index  = self.indexer()

    def store_system_message(self, message: str, *args, **kwargs):
        val = f"[SYSTEM::INSTRUCTION] <<<\n{str(message)}\n>>>\n"
        self.store(val, *args, **kwargs)

    def store_file(self, file_path: str, *args, **kwargs):
        # read in file
        with open(file_path, 'r') as file:
            content = file.read()
        val = f"[DATA::{file_path}] <<<\n{str(content)}\n>>>\n"
        self.store(val, *args, **kwargs)

    def commit(self, formatter: Optional[Callable] = None):
        if self.file_link is not None:
            # if file extension is .py, then format code
            format_ = formatter
            formatter = CodeFormatter() if format_ is None and self.file_link.endswith('.py') else formatter
            val = self.value
            if formatter is not None:
                val = formatter(val)
            with open(self.file_link, 'w') as file:
                file.write(str(val))

    def save(self, path: str, replace: bool = False) -> Symbol:
        return Symbol(self._memory).save(path, replace=replace)

    def index(self, file_path: str):
        return self._index(file_path)

    def forward(self, query: str, *args, **kwargs):
        query = self._to_symbol(query)
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        # append to string to memory
        val = str(f"[USER::{timestamp}] <<<\n{str(query)}\n>>>\n")
        self.store(val, *args, **kwargs)
        history = Symbol(f'[HISTORY] <<<\n{self._memory}\n>>>\n')
        if 'payload' in kwargs:
            history =  f'{history}\n{kwargs["payload"]}'
            del kwargs['payload']
        res = self.recall(query, payload=history, *args, **kwargs)
        self.value = res.value # save last response
        val = str(f"[ASSISTANT::{timestamp}] <<<\n{str(res)}\n>>>\n")
        self.store(val, *args, **kwargs)
        if self.auto_print:
            print(res)
        return res
