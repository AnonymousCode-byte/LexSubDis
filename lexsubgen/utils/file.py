'''
该文件提供了一系列实用的文件处理和数据加载函数，方便在词汇替换生成任务中进行各种操作。
'''
import gzip
import importlib
import json
import pkgutil
import shutil
import tarfile
import zipfile
import requests
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple, Union, Any, NoReturn
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import wget # webget下载文件
from _jsonnet import evaluate_file
from tqdm import tqdm
from lexsubgen.utils.register import CACHE_DIR, memory  # 缓存目录+缓存对象

# 根据文件的后缀名，将不同类型的压缩文件（如 .zip, .tgz, .gz 等）解压到指定的目标文件夹。
def extract_archive(arch_path: Union[str, Path], dest: str):
    """
    Extracts archive into a given folder.

    Args:
        arch_path: path to archive file.
            Could be given as string or `pathlib.Path`.
        dest: path to destination folder.
    """
    arch_path = Path(arch_path)
    file_suffixes = arch_path.suffixes
    outer_suffix = file_suffixes[-1]
    inner_suffix = file_suffixes[-2] if len(file_suffixes) > 1 else ""
    if outer_suffix == ".zip":
        with zipfile.ZipFile(arch_path, "r") as fp:
            fp.extractall(path=dest)
    elif outer_suffix == ".tgz" or (outer_suffix == ".gz" and inner_suffix == ".tar"):
        with tarfile.open(arch_path, "r:gz") as tar:
            dirs = [member for member in tar.getmembers()]
            tar.extractall(path=dest, members=dirs)
    elif outer_suffix == ".gz":
        with gzip.open(arch_path, "rb") as gz:
            with open(
                arch_path.parent / (arch_path.stem + inner_suffix), "wb"
            ) as uncomp:
                shutil.copyfileobj(gz, uncomp)

# 加载单词频率数据。如果数据文件不存在，则从指定的 URL 下载，并将其存储在缓存目录中。最后返回一个字典，包含单词及其对应的频率。
@memory.cache
def load_word2freq() -> Dict[str, int]:
    """
    Loads word frequencies data.

    Returns:
        mapping from words to their counts
    """
    freq_words_path = CACHE_DIR / "resources" / "count_1w.txt"
    if not freq_words_path.exists():
        freq_words_path.mkdir(freq_words_path.parent, parents=True, exist_ok=True)
        urlretrieve("http://norvig.com/ngrams/count_1w.txt", filename=freq_words_path)

    frequency_words = pd.read_csv(
        freq_words_path, sep="\t", header=None, names=["word", "count"]
    )
    total = len(frequency_words)
    gen = zip(frequency_words["word"], frequency_words["count"])
    word2freq = {
        word: count
        for word, count in tqdm(gen, desc="Loading frequency words", total=total)
    }
    return word2freq

# 从指定的 URL 下载嵌入文件，并将其解压到指定的目标目录。下载完成后，删除原始的压缩文件。
# 嵌入文件干啥的？
def download_embeddings(url: str, dest: Union[str, Path]):
    dest_path = Path(dest)
    if not dest_path.exists():
        dest_path.mkdir(parents=True)
    file_name = wget.download(url, out=str(dest))
    extract_archive(file_name, dest)
    file_path = Path(file_name)
    file_path.unlink()

# 从指定的文件中加载嵌入矩阵。文件的第一行应包含词汇表大小和嵌入向量的维度，后续行应包含单词及其对应的嵌入向量。返回嵌入矩阵、单词到索引的映射和索引到单词的映射。
@memory.cache
def get_emb_matrix(file_name: str) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Loads embedding matrix from a given file. The embeddings should be stored
    in the following format. First row of data consists of two values: vocabulary size
    and size of the embeddings. Each next row contains word represented as a string and
    sequence of embedding vector values.

    Args:
        file_name: path to the file containing embeddings.

    Returns:
        `numpy.ndarray` - embedding matrix, and two representations of vocabulary:
        mapping from words to their indexes and list of words.
    """
    count = 0
    word2id = {}
    vocab = {}
    embeddings_list = []
    with open(file_name) as f:
        vocab_size, embedding_size = f.readline().split()
        vocab_size, embedding_size = int(vocab_size), int(embedding_size)
        for idx in range(vocab_size):
            word, values = f.readline().split(maxsplit=1)
            values = values.split()
            if len(values) == embedding_size:
                word2id[word] = count
                vocab[count] = word
                embeddings_list.append(np.array([float(val) for val in values]))
                count += 1
            else:
                raise ValueError(
                    f"Wrong size of word embedding {len(values)}. "
                    f"Expected {embedding_size} elements."
                )
    return np.stack(embeddings_list, axis=0), word2id, vocab

# 从 Google Drive 下载大文件。通过处理 Google Drive 的下载警告，确保能够正确下载文件。
def download_large_gdrive_file(
    file_id: str, dst: str,
    gdrive_url: str = "https://docs.google.com/uc?export=download",
) -> NoReturn:

    with requests.Session() as s:
        response = s.get(
            gdrive_url,
            params={"id": file_id},
            stream=True,
        )

        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            response = s.get(
                gdrive_url,
                params={"id": file_id, "confirm": token},
                stream=True,
            )

        with open(dst, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)

# 将 Python 对象以 JSON 格式保存到指定的文件中。
# JSON无法序列化自定义对象
def dump_json(path: Union[str, Path], object: Any):
    """
    Saves object in the given directory in JSON format

    Args:
        path: This directory must have already been created.
            Could be a string or `pathlib.Path` object.
        object: data to store.
    """
    with Path(path).open("w") as fp:
        json.dump(object, fp, indent=4)

# import json
# from json import JSONEncoder
#
# class CustomEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, SubstituteGenerator):
#             # 提取可序列化的配置信息
#             return {"class": obj.__class__.__name__, "config": obj.get_config()}
#         return super().default(obj)
#
# def dump_json(path: Union[str, Path], object: Any):
#     with Path(path).open("w") as fp:
#         json.dump(object, fp, indent=4, cls=CustomEncoder)


# 创建实验运行目录。如果 force 为 True，则会删除已存在的目录并重新创建。
# 生成的目录中不能有：windows，会非法
# def create_run_dir(run_dir: Union[str, Path], force: bool = False):
#     """
#     Creates experiment (run) directory. Saves experiment configuration file.
#     If `force` is true, will overwrite data in an existing directory.
#
#     Args:
#         run_dir: path to a directory where to store experiment results.
#             Could be a string or `pathlib.Path` object.
#         force: whether to overwrite data in an existing directory.
#
#     """
#     run_path = Path(run_dir)
#     if run_path.exists() and force:
#         shutil.rmtree(run_path)
#     run_path.mkdir(parents=True, exist_ok=False)
from datetime import datetime

def create_run_dir(run_path: Path, force: bool = False):
    if run_path.exists() and not force:
        raise FileExistsError(f"Directory '{run_path}' already exists.")
    # 修改时间戳格式
    # timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # run_path = run_path / timestamp
    run_path.mkdir(parents=True, exist_ok=False)
    # path:默认为 False。当设置为 True 时，如果要创建的目录的父目录不存在，会递归地创建所有必要的父目录。

# 从给定的模块路径导入子模块。可以递归地导入子模块，同时会跳过第三方包。
def import_submodules(module: Union[str, ModuleType], recursive: bool = True):
    """
    Imports submodules from a given path. This could also be done recursively.

    Args:
        module: module path.
        recursive: whether to load submodules recursively (default: True).

    """
    importlib.invalidate_caches()

    if isinstance(module, str):
        module = importlib.import_module(module)

    module_path = module.__path__
    first_path = module_path[0] if module_path else ""

    for module_finder, name, is_pkg in pkgutil.walk_packages(module_path):
        if first_path and module_finder.path != first_path:
            # skip 3-rd party packages
            continue
        submodule_name = module.__name__ + "." + name
        if recursive and is_pkg:
            import_submodules(submodule_name, recursive)

# 检查指定的 .jsonnet 文件是否有效。通过尝试评估文件内容来判断文件的有效性。
def is_valid_jsonnet(file_path: Union[str, Path]) -> bool:
    file_path = Path(file_path)
    assert file_path.suffix == ".jsonnet"

    try:
        evaluate_file(str(file_path))
        return True
    except Exception as e:
        return False
