# Предсказание поломок станка

Учебный пет-проект по ML на табличных данных. Предсказываю аварийный отказ промышленного оборудования (`Machine failure`: 0 или 1) по показаниям датчиков.

**Стек:** Python, Pandas, Scikit-learn, Matplotlib/Seaborn.

---

### Что сделано:
* **Датасет:** [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Из-за сильного дисбаланса классов (аварий всего ~3.4%) в качестве главной метрики взят ROC-AUC
* **Предобработка:** 
  * Удалила ID и колонки с конкретными видами поломок (`TWF`, `HDF` и др.), чтобы исключить утечку данных
  * Для Type применила OrdinalEncoder
* **Валидация:** Stratified Train/Test split (80/20) + 5-Fold кросс-валидация на трейне
* **Модели:** 
  * Бейзлайн: LogisticRegression (с StandardScaler)
  * Финальная модель: RandomForestClassifier с подбором гиперпараметров через RandomizedSearchCV

---

### Результаты:
* **Baseline (LogReg) ROC-AUC:** `0.895`
* **Random Forest ROC-AUC на тесте:** `0.966`

**Топ датчиков по важности:** крутящий момент (`Torque`), износ инструмента (`Tool wear`) и скорость вращения (`Rotational speed`)
