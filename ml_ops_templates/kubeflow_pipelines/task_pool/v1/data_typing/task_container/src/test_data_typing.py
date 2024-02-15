import pandas as pd
import json

from data_typing import DataTyping
from typing import Dict, List

ANALYTICAL_DATA_TYPES: Dict[str, List[str]] = dict(categorical=["type",
                                                                "year",
                                                                "region",
                                                                "Unnamed: 0"
                                                                ],
                                                   ordinal=[],
                                                   continuous=["AveragePrice",
                                                               "Total Volume",
                                                               "4046",
                                                               "4225",
                                                               "4770",
                                                               "Total Bags",
                                                               "Small Bags",
                                                               "Large Bags",
                                                               "XLarge Bags"
                                                               ],
                                                   date=["Date"],
                                                   id_text=[]
                                                   )

_df: pd.DataFrame = pd.read_csv('/Users/giannibalistreri/Downloads/avocado_typed.csv')
_data_typing: DataTyping = DataTyping(df=_df,
                                      feature_names=_df.columns.tolist(),
                                      analytical_data_types=ANALYTICAL_DATA_TYPES,
                                      missing_value_features=None,
                                      data_types_config=None
                                      )
_data_typing.main()
_data_typing.df.to_json('/Users/giannibalistreri/Downloads/avocado_typed_2.json')
