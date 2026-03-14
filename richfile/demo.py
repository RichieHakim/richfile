from typing import Optional, Union

from pathlib import Path

import richfile as rf

class RichFile_data(rf.RichFile):
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        check: Optional[bool] = True,
        safe_save: Optional[bool] = True,
    ):
        super().__init__(path=path, check=check, safe_save=safe_save)

        type_dicts = []
        
        ## NUMPY ARRAY
        try:
            import numpy as np

            def save_npy_array(
                obj: np.ndarray,
                path: Union[str, Path],
                **kwargs,
            ) -> None:
                """
                Saves a NumPy array to the given path.
                """
                np.save(path, obj, **kwargs)

            def load_npy_array(
                path: Union[str, Path],
                **kwargs,
            ) -> np.ndarray:
                """
                Loads an array from the given path.
                """    
                return np.load(path, **kwargs)
            
            def save_npy_scalar(
                obj: np.number,
                path: Union[str, Path],
                **kwargs,
            ) -> None:
                """
                Saves a NumPy scalar to the given path.
                """
                np.save(path, np.array(obj), **kwargs)

            def load_npy_scalar(
                path: Union[str, Path],
                **kwargs,
            ) -> np.number:
                """
                Loads a NumPy scalar from the given path.
                """
                return np.load(path, **kwargs).item()

            type_dicts.extend([
                {
                    "type_name":          "numpy_array",
                    "function_load":      load_npy_array,
                    "function_save":      save_npy_array,
                    "object_class":       np.ndarray,
                    "suffix":             "npy",
                    "library":            "numpy",
                    "versions_supported": [],
                },
                {
                    "type_name":          "numpy_scalar",
                    "function_load":      load_npy_scalar,
                    "function_save":      save_npy_scalar,
                    "object_class":       np.number,
                    "suffix":             "npy",
                    "library":            "numpy",
                    "versions_supported": [],
                },
            ])
        except ImportError:
            pass
        

        ## SCIPY SPARSE MATRIX
        try:
            import scipy.sparse

            def save_sparse_array(
                obj: scipy.sparse.spmatrix,
                path: Union[str, Path],
                **kwargs,
            ) -> None:
                """
                Saves a SciPy sparse matrix to the given path.
                """
                scipy.sparse.save_npz(path, obj, **kwargs)

            def load_sparse_array(
                path: Union[str, Path],
                **kwargs,
            ) -> scipy.sparse.csr_matrix:
                """
                Loads a sparse array from the given path.
                """        
                return scipy.sparse.load_npz(path, **kwargs)
            
            type_dicts.extend([
                {
                    "type_name":          "scipy_sparse_array",
                    "function_load":      load_sparse_array,
                    "function_save":      save_sparse_array,
                    "object_class":       scipy.sparse.spmatrix,
                    "suffix":             "npz",
                    "library":            "scipy",
                    "versions_supported": [],
                },
            ])
        except ImportError:
            pass
        
        ## TORCH TENSOR
        try:
            import torch

            def save_torch_tensor(
                obj: torch.Tensor,
                path: Union[str, Path],
                **kwargs,
            ) -> None:
                """
                Saves a PyTorch tensor to the given path as a NumPy array.
                """
                np.save(path, obj.detach().cpu().numpy(), **kwargs)

            def load_torch_tensor(
                path: Union[str, Path],
                **kwargs,
            ) -> torch.Tensor:
                """
                Loads a PyTorch tensor from the given path.
                """
                return torch.from_numpy(np.load(path, **kwargs))
            
            type_dicts.extend([
                {
                    "type_name":          "torch_tensor",
                    "function_load":      load_torch_tensor,
                    "function_save":      save_torch_tensor,
                    "object_class":       torch.Tensor,
                    "suffix":             "npy",
                    "library":            "torch",
                    "versions_supported": [],
                },
            ])
        except ImportError:
            pass

        ## PANDAS DATAFRAME
        try:
            import pandas as pd
            
            def save_pandas_dataframe(
                obj: pd.DataFrame,
                path: Union[str, Path],
                **kwargs,
            ) -> None:
                """
                Saves a Pandas DataFrame to the given path.
                """
                ## Save as a CSV file
                obj.to_csv(path, index=True, **kwargs)

            def load_pandas_dataframe(
                path: Union[str, Path],
                **kwargs,
            ) -> pd.DataFrame:
                """
                Loads a Pandas DataFrame from the given path.
                """
                ## Load as a CSV file
                return pd.read_csv(path, index_col=0, **kwargs)
            
            type_dicts.extend([
                {
                    "type_name":          "pandas_dataframe",
                    "function_load":      load_pandas_dataframe,
                    "function_save":      save_pandas_dataframe,
                    "object_class":       pd.DataFrame,
                    "suffix":             "csv",
                    "library":            "pandas",
                    "versions_supported": [],
                },
            ])
        except ImportError:
            pass

        ## JSON DICT
        import collections
        import json

        def save_json_dict(
            obj: collections.UserDict,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a dictionary to the given path.
            """
            with open(path, 'w') as f:
                json.dump(dict(obj), f, **kwargs)

        def load_json_dict(
            path: Union[str, Path],
            **kwargs,
        ) -> collections.UserDict:
            """
            Loads a dictionary from the given path.
            """
            with open(path, 'r') as f:
                return JSON_Dict(json.load(f, **kwargs))

        ## JSON LIST   
        def save_json_list(
            obj: collections.UserList,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a list to the given path.
            """
            with open(path, 'w') as f:
                json.dump(list(obj), f, **kwargs)

        def load_json_list(
            path: Union[str, Path],
            **kwargs,
        ) -> collections.UserList:
            """
            Loads a list from the given path.
            """
            with open(path, 'r') as f:
                return JSON_List(json.load(f, **kwargs))
            
        type_dicts.extend([
            {
                "type_name":          "json_dict",
                "function_load":      load_json_dict,
                "function_save":      save_json_dict,
                "object_class":       JSON_Dict,
                "suffix":             "json",
                "library":            "python",
                "versions_supported": [],
            },
            {
                "type_name":          "json_list",
                "function_load":      load_json_list,
                "function_save":      save_json_list,
                "object_class":       JSON_List,
                "suffix":             "json",
                "library":            "python",
                "versions_supported": [],
            },
        ])

        [self.register_type_from_dict(d) for d in type_dicts]
        

######################################
######## CUSTOM DATA CLASSES #########
######################################

class JSON_Dict(dict):
    def __init__(self, *args, **kwargs):
        super(JSON_Dict, self).__init__(*args, **kwargs)
class JSON_List(list):
    def __init__(self, *args, **kwargs):
        super(JSON_List, self).__init__(*args, **kwargs)
