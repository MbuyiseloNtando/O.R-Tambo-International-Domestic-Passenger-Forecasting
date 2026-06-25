{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM30qdGPGCVJxnqKg5nPvzf",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MbuyiseloNtando/O.R-Tambo-International-Domestic-Passenger-Forecasting/blob/main/XGboostModel.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TUXLezkEQYv0"
      },
      "outputs": [],
      "source": [
        "#Importing manipulation and Visualion libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from pandas.plotting import autocorrelation_plot\n",
        "from statsmodels.graphics.tsaplots import plot_pacf\n",
        "\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Set the overall dark background (fixes text/axes colors)\n",
        "plt.style.use('dark_background')\n",
        "\n",
        "# 2. Define and apply  specific color palette\n",
        "sns.set_palette(palette='Paired')\n",
        "\n",
        "#Add a Seaborn style for better spacing/grid\n",
        "sns.set_style(\"ticks\")"
      ],
      "metadata": {
        "id": "NrxKfS6VQcgU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install skforecast"
      ],
      "metadata": {
        "id": "mu557Z8BXt-C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#importing our dataset\n",
        "data = 'https://raw.githubusercontent.com/MbuyiseloNtando/O.R-Tambo-International-Domestic-Passenger-Forecasting/refs/heads/main/Cleaned%20and%20Transformed%20Data/Complete%20Dataset.csv'\n",
        "df = pd.read_csv(data)\n",
        "df.head(10)"
      ],
      "metadata": {
        "id": "qd5MSu1xQdRJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0baf5017-2b6c-4a8a-83cc-45a64ec2a755"
      },
      "outputs": [],
      "source": [
        "df = df[['Date', 'Passengers', 'tourists.tourists', 'CPI.CPI', 'GDP.GDP', 'Population.Population',\n",
        "         'Trade_balance.Trade balance (Rands)', 'Crude_oil.Crude Oil', 'Unemployment_rate.Unemployment']]\n",
        "df.head(10)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Remove a know duplicated rows\n",
        "df.drop(df.index[4], inplace=True)\n",
        "df = df[:-1]"
      ],
      "metadata": {
        "id": "0eji_ieYFrnu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.duplicated().sum()"
      ],
      "metadata": {
        "id": "saLW7GMaHLWO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df[df.duplicated()]"
      ],
      "metadata": {
        "id": "_t-8o7yPHQLc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "df04798f-477c-4c49-84bc-f573cc2a6f54"
      },
      "outputs": [],
      "source": [
        "#Setting the date as index\n",
        "df['Date'] = pd.to_datetime(df['Date'])\n",
        "df.set_index('Date', inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6312d022-81aa-46e7-bc00-43e999258bcb"
      },
      "outputs": [],
      "source": [
        "#dding covi19 dummy\n",
        "df['covid_flag'] = ((df.index >='2020-03-01') & (df.index < '2021-08-01')).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns"
      ],
      "metadata": {
        "id": "WxB_VBolwPME"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6943f4bf-e807-407d-9cae-f1611f576961"
      },
      "outputs": [],
      "source": [
        "df.rename(columns ={'Crude_oil.Crude Oil' : 'crude_oil',\n",
        "                  'tourists.tourists' : 'tourists',\n",
        "                  'CPI.CPI' : 'cpi',\n",
        "                  'GDP.GDP' : 'gdp',\n",
        "                   'Population.Population' : 'population',\n",
        "                   'Trade_balance.Trade balance (Rands)': 'Trade_balance',\n",
        "                  'Unemployment_rate.Unemployment' : 'unemployment_rate'\n",
        "                    },\n",
        "          inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "id": "noyG3Tjz0bIQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "39b6c4e8-a965-4e92-9da5-3fc1878bce61"
      },
      "outputs": [],
      "source": [
        "#interpolting gdp nd popultion columns\n",
        "col = ['gdp', 'population']\n",
        "df[col] = df[col].interpolate(method='linear')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5b287c38-98e1-4318-b1f3-051ef67cbee7"
      },
      "outputs": [],
      "source": [
        "#front filling tourists and cpi\n",
        "df['tourists'] = df.tourists.ffill()\n",
        "df['cpi'] = df.cpi.ffill()\n",
        "df['unemployment_rate'] = df.unemployment_rate.ffill()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "efa2dba5-0ee1-4155-a57e-d5664e501a2b"
      },
      "outputs": [],
      "source": [
        "df['gdp_per_capta'] = df['gdp']/df['population']\n",
        "df.drop(['population', 'gdp'], axis=1,\n",
        "        inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9e5f6ec4-ab62-401c-bd68-7732c3ea479d"
      },
      "outputs": [],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4d408f41-d0d2-44a7-886d-678a56a70c4e"
      },
      "outputs": [],
      "source": [
        "df.isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7b1153f7-b180-47bc-afb9-cabc7de4f02d"
      },
      "outputs": [],
      "source": [
        "df['month'] = df.index.month\n",
        "df['year'] = df.index.year"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0b0a85aa-1439-4746-af38-353a7a48073f"
      },
      "outputs": [],
      "source": [
        "df['covid_flag'].value_counts()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "113598f3-8f0f-449b-964e-00d4aa299563"
      },
      "outputs": [],
      "source": [
        "df.isna().sum()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df[df.index.duplicated()]"
      ],
      "metadata": {
        "id": "DYpHZDSEYJeD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sns.lineplot(data=df, x=df.index, y='Passengers')\n",
        "plt.title('Monthly Passengers')"
      ],
      "metadata": {
        "id": "_bJt3stvQuxn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_xg = df.copy()\n",
        "df_xg.head()"
      ],
      "metadata": {
        "id": "zSgegcA3Qzxv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_xg1 = df_xg.copy()\n",
        "df_xg1=df_xg1.asfreq('MS')"
      ],
      "metadata": {
        "id": "Y6DSqZFVRCrE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_xg1['month'] = df.index.month\n",
        "df_xg1['year'] = df.index.year"
      ],
      "metadata": {
        "id": "oFtI6yLadp8M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_xg1.columns"
      ],
      "metadata": {
        "id": "FgQSt8l3d0Vf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from statsmodels.tsa.seasonal import seasonal_decompose\n",
        "\n",
        "# Verifying seasonality\n",
        "season = seasonal_decompose(df['Passengers'],\n",
        "                            model='additive', period=12)\n",
        "season.plot()"
      ],
      "metadata": {
        "id": "RXvAhpjLebFN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), sharex=True)\n",
        "autocorrelation_plot(df_xg1['Passengers'], ax=axs[0])\n",
        "plot_pacf(df_xg1['Passengers'], ax=axs[1])"
      ],
      "metadata": {
        "id": "YF4bj-2Oe1dU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#train and test sets\n",
        "def train_test_split(df, test_size):\n",
        "  test_index = int(len(df_xg1) * (1 - test_size))\n",
        "  train, test = df[:test_index], df[test_index:]\n",
        "  return train, test\n",
        "xg1_train, xg1_test = train_test_split(df_xg1, 0.2)"
      ],
      "metadata": {
        "id": "LKtEaQNSZgJJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "len(df_xg1), len(xg1_train), len(xg1_test)"
      ],
      "metadata": {
        "id": "8KZDgeKMZyrY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9560737b-aa6d-4623-b9e5-c57c317206d7"
      },
      "outputs": [],
      "source": [
        "from skforecast.preprocessing import RollingFeatures\n",
        "window_features = RollingFeatures(\n",
        "    stats=[\"mean\", \"std\"],\n",
        "    window_sizes=[24, 24]\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6dd2a783-0ef2-4c33-ab07-b9072ca88916"
      },
      "outputs": [],
      "source": [
        "from xgboost import XGBRegressor\n",
        "from skforecast.recursive import ForecasterRecursive\n",
        "\n",
        "forecaster = ForecasterRecursive(\n",
        "                estimator       = XGBRegressor(enable_categorical=True),\n",
        "                lags            = 25,\n",
        "                window_features = window_features\n",
        "\n",
        "             )\n",
        "\n",
        "# Train forecaster\n",
        "# ==============================================================================\n",
        "forecaster.fit(y=xg1_train['Passengers'])\n",
        "forecaster\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0f815097-0daf-46d6-8fb4-ce27fe4a47f8"
      },
      "outputs": [],
      "source": [
        "pred = forecaster.predict(steps=33)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ad3d5a2a-ac9b-48ac-9c83-f049d36eec21"
      },
      "outputs": [],
      "source": [
        "pred.plot(legend=True)\n",
        "xg1_test['Passengers'].plot(legend=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d1b7f067-6c9f-42e2-851d-668af7c99b34"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
        "mean_absolute_error(y_true=xg1_test['Passengers'],\n",
        "                   y_pred=pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "885b0351-c903-45d6-b52f-b3f4d3f1fc1b"
      },
      "outputs": [],
      "source": [
        "#Back seating\n",
        "from skforecast.model_selection import TimeSeriesFold\n",
        "from skforecast.model_selection import backtesting_forecaster\n",
        "\n",
        "cv = TimeSeriesFold(\n",
        "        steps              = 12,\n",
        "        initial_train_size = 120,\n",
        "        refit              = True,\n",
        ")\n",
        "\n",
        "def weighted_absolute_percentage_error(y_true, y_pred):\n",
        "    \"\"\"\n",
        "    Custom WAPE metric for data with zeros.\n",
        "    \"\"\"\n",
        "    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))\n",
        "\n",
        "metric, predictions = backtesting_forecaster(\n",
        "    forecaster = forecaster,\n",
        "    y          = xg1_train['Passengers'],\n",
        "    cv         = cv,\n",
        "    metric     = ['mean_squared_error', weighted_absolute_percentage_error, 'mean_absolute_error']\n",
        ")\n",
        "\n",
        "metric"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fcd30168-a35a-49d2-bbb4-9a66b9fe862c"
      },
      "outputs": [],
      "source": [
        "predictions['pred'].plot()\n",
        "xg1_train['Passengers'].plot()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e8f98409-56a3-42ae-b3df-81099f40a486"
      },
      "source": [
        "#### **Exogeneous Variables**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "14be89bf-7161-46f6-b706-1299973258ce"
      },
      "outputs": [],
      "source": [
        "import holidays\n",
        "\n",
        "daily_rng = pd.date_range(start='2012-04-01', end='2025-12-01', freq='D')\n",
        "us_holidays = holidays.ZA()\n",
        "\n",
        "# 2. Map holidays to the daily range\n",
        "df_daily = pd.DataFrame(index=daily_rng)\n",
        "df_daily['is_holiday'] = df_daily.index.map(lambda x: 1 if x in us_holidays else 0)\n",
        "\n",
        "# 3. Resample to Month Start (\"MS\") frequency\n",
        "# Summing gives \"Holiday Count\" per month\n",
        "df_monthly_holidays = df_daily['is_holiday'].resample('MS').sum().to_frame()\n",
        "df_monthly_holidays['Month'] = df.index.month\n",
        "df_monthly_holidays['Year'] = df.index.year\n",
        "\n",
        "#Covid19 dummy\n",
        "target_dates = df_monthly_holidays.index[(df_monthly_holidays.index >='2020-03-01') & (df_monthly_holidays.index < '2021-08-01')]\n",
        "target_dates2 = df_monthly_holidays.index[(df_monthly_holidays.index >='2013-03-01') & (df_monthly_holidays.index < '2014-02-01')]\n",
        "\n",
        "# Create the dummy column\n",
        "df_monthly_holidays['covid_binary'] = df_monthly_holidays.index.isin(target_dates).astype(int)\n",
        "df_monthly_holidays['bankrupsy_binary'] = df_monthly_holidays.index.isin(target_dates2).astype(int)\n",
        "\n",
        "df_monthly_holidays.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "91162ec1-14d2-463a-b79a-98fc4a1a1b96"
      },
      "outputs": [],
      "source": [
        "df_monthly_holidays['covid_binary'].value_counts()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d7b29039-b2b9-4b20-b615-5c678ce6e71c"
      },
      "outputs": [],
      "source": [
        "df_monthly_holidays['bankrupsy_binary'].value_counts()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6d529a65-facc-4d21-aeb8-3b6352081000"
      },
      "outputs": [],
      "source": [
        "merged_df = pd.merge(df_xg1, df_monthly_holidays, left_index=True, right_index=True, how='inner')\n",
        "merged_df=merged_df.asfreq('MS')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "69f5cbb7-0b0a-4c82-ba81-f2170ec07502"
      },
      "outputs": [],
      "source": [
        "from sklearn.preprocessing import FunctionTransformer\n",
        "\n",
        "# Cyclical encoding with sine/cosine transformation\n",
        "def sin_transformer(period):\n",
        "\treturn FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))\n",
        "\n",
        "def cos_transformer(period):\n",
        "\treturn FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))\n",
        "\n",
        "data_encoded_sin_cos = merged_df.copy()\n",
        "data_encoded_sin_cos[\"month_sin\"] = sin_transformer(12).fit_transform(merged_df['Month'])\n",
        "data_encoded_sin_cos[\"month_cos\"] = cos_transformer(12).fit_transform(merged_df['Month'])\n",
        "data_encoded_sin_cos.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "acacdeb0-4f28-4b63-a7bb-2928fa2d3d2c"
      },
      "outputs": [],
      "source": [
        "fig, ax = plt.subplots(figsize=(4., 3.5))\n",
        "sp = ax.scatter(\n",
        "        data_encoded_sin_cos[\"month_sin\"],\n",
        "        data_encoded_sin_cos[\"month_cos\"],\n",
        "        c=data_encoded_sin_cos[\"Month\"],\n",
        "        cmap='viridis'\n",
        "     )\n",
        "ax.set(\n",
        "    xlabel=\"sin(month)\",\n",
        "    ylabel=\"cos(month)\",\n",
        ")\n",
        "_ = fig.colorbar(sp)\n",
        "data_encoded_sin_cos = data_encoded_sin_cos.drop(columns='Month')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "37fb8bbf-1687-42fd-8d35-0dbdb19434d0"
      },
      "outputs": [],
      "source": [
        "# Assuming 'df' has a DatetimeIndex with 'MS' frequency\n",
        "#data_encoded_sin_cos['oil_roll_mean_3M'] = data_encoded_sin_cos['crude_oil'].rolling(window=3).mean()\n",
        "#data_encoded_sin_cos['oil_roll_mean_1Y'] = data_encoded_sin_cos['crude_oil'].rolling(window=12).mean()\n",
        "data_encoded_sin_cos['tourists_roll_mean_3M'] = data_encoded_sin_cos['tourists'].rolling(window=3).mean()\n",
        "data_encoded_sin_cos['tourists_roll_mean_1Y'] = data_encoded_sin_cos['tourists'].rolling(window=12).mean()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "69de3f2b-103f-468d-95f1-0c8f6bddbdc8"
      },
      "outputs": [],
      "source": [
        "#train and test sets\n",
        "df_train_xg2 = data_encoded_sin_cos[:132]\n",
        "df_test_xg2 = data_encoded_sin_cos[132:]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ff97391c-cec1-4f7e-abaa-3812c76d099c"
      },
      "outputs": [],
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "scaler = StandardScaler()\n",
        "# Fits to data and transforms it in one step\n",
        "scaled_data = scaler.fit_transform(df_train_xg2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9379196a-56f0-4fd1-be7f-031412b9c0ef"
      },
      "outputs": [],
      "source": [
        "df_train_xg2.columns"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "06f88e06-6710-4f49-bf61-7ea264366846"
      },
      "outputs": [],
      "source": [
        "exog_features = ['bankrupsy_binary','is_holiday', 'covid_binary',\n",
        "                 'month_sin', 'month_cos', 'tourists_roll_mean_3M','tourists_roll_mean_1Y' ]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "906c6230-b56d-4da7-bd6e-ca60aafcff68"
      },
      "outputs": [],
      "source": [
        "from xgboost import XGBRegressor\n",
        "from skforecast.model_selection import backtesting_forecaster\n",
        "\n",
        "def weighted_absolute_percentage_error(y_true, y_pred):\n",
        "    \"\"\"\n",
        "    Custom WAPE metric for data with zeros.\n",
        "    \"\"\"\n",
        "    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))\n",
        "\n",
        "metric, predictions = backtesting_forecaster(\n",
        "                forecaster= forecaster,\n",
        "                y = df_train_xg2['Passengers'],\n",
        "                exog = data_encoded_sin_cos[exog_features],\n",
        "                cv = cv,\n",
        "                metric = [weighted_absolute_percentage_error, 'mean_absolute_error']\n",
        "             )\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "45a3829a-af63-4fb3-ae92-6062af564633"
      },
      "outputs": [],
      "source": [
        "metric"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d3dcc53c-1c1e-4441-aa35-ad79f2d83181"
      },
      "outputs": [],
      "source": [
        "predictions['pred'].plot()\n",
        "df_train_xg2['Passengers'].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4ce49762-f6dc-4682-889a-b7fcb2dc51f8"
      },
      "outputs": [],
      "source": [
        "predictions['pred'].plot()\n",
        "df_train_xg2['Passengers'].plot()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9074b2d0-91c2-477e-af63-d6ca474ee92a"
      },
      "outputs": [],
      "source": [
        "# Hyperparameters search\n",
        "from skforecast.model_selection import bayesian_search_forecaster\n",
        "\n",
        "# Lags grid\n",
        "lags_grid = [48, 72, [1, 2, 3, 23, 24, 25]]\n",
        "\n",
        "# Estimator hyperparameters search space\n",
        "def search_space(trial):\n",
        "    search_space  = {\n",
        "        'n_estimators'    : trial.suggest_int('n_estimators', 400, 1200, step=100),\n",
        "        'max_depth'       : trial.suggest_int('max_depth', 3, 10, step=1),\n",
        "        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 1),\n",
        "        'subsample'       : trial.suggest_float('subsample', 0.1, 1),\n",
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),\n",
        "        'gamma'           : trial.suggest_float('gamma', 0, 1),\n",
        "        'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),\n",
        "        'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),\n",
        "        'lags'            : trial.suggest_categorical('lags', lags_grid)\n",
        "    }\n",
        "    return search_space\n",
        "\n",
        "# Folds\n",
        "cv_searh = TimeSeriesFold(\n",
        "                steps              = 12,\n",
        "                initial_train_size = 120,\n",
        "                refit              = True,\n",
        "            )\n",
        "def weighted_absolute_percentage_error(y_true, y_pred):\n",
        "    \"\"\"\n",
        "    Custom WAPE metric for data with zeros.\n",
        "    \"\"\"\n",
        "    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))\n",
        "\n",
        "results_search, frozen_trial = bayesian_search_forecaster(\n",
        "                                    forecaster= forecaster,\n",
        "                                    y = df_train_xg2['Passengers'],\n",
        "                                    exog = df_train_xg2[exog_features],\n",
        "                                    search_space = search_space,\n",
        "                                    cv           = cv_searh,\n",
        "                                    metric       = ['mean_absolute_error', weighted_absolute_percentage_error],\n",
        "                                    n_trials     = 20\n",
        "                                )\n",
        "best_params = results_search['params'].iat[0]\n",
        "best_lags = results_search['lags'].iat[0]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "22e3f60a-aba0-4d28-b2c5-ac7419eec00d"
      },
      "outputs": [],
      "source": [
        "forecaster"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f3ab1ef0-d1fc-48a0-8a3d-3174cd8540fe"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error\n",
        "from math import sqrt\n",
        "# Backtesting model with exogenous variables on test data\n",
        "\n",
        "def weighted_absolute_percentage_error(y_true, y_pred):\n",
        "    \"\"\"\n",
        "    Custom WAPE metric for data with zeros.\n",
        "    \"\"\"\n",
        "    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))\n",
        "def root_mean_squared_error(y_true, y_pred):\n",
        "    return np.sqrt(np.mean(np.square(y_true - y_pred)))\n",
        "\n",
        "metric, predictions = backtesting_forecaster(\n",
        "                            forecaster = forecaster,\n",
        "                            y          = data_encoded_sin_cos['Passengers'],\n",
        "                            exog = data_encoded_sin_cos[exog_features],\n",
        "                            cv         = cv,\n",
        "                            metric     = [weighted_absolute_percentage_error, root_mean_squared_error,\n",
        "                                          'mean_absolute_error', r2_score]\n",
        "                       )\n",
        "display(metric)\n",
        "predictions.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9ac09ebb-0297-48f6-b3de-16075f743a44"
      },
      "outputs": [],
      "source": [
        "predictions['pred'].plot(legend = True)\n",
        "df_test_xg2['Passengers'].plot(legend = True)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle"
      ],
      "metadata": {
        "id": "EHLXfA7Rh8aO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "file_name = 'trained_OR-Forecast_model.sav'\n",
        "pickle.dump(forecaster, open(file_name, 'wb'))\n",
        "#"
      ],
      "metadata": {
        "id": "LV7pSDIxkYxS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}