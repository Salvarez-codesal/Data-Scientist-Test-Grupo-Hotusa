# Grupo Hotusa - Data Scientist Test - Soraya Alvarez Codesal

El director de Revenue Management nos comenta que las cancelaciones tienen un impacto negativo en los resultados del hotel y se pregunta si con todo esto del Machine Learning y la Inteligencia Artificial seríamos capaces de predecir qué reservas van a cancelar y cuáles no. 

El director quiere poner en marcha este modelo para poder realizar overbooking (llenar el hotel por encima del inventario total disponible con el objetivo de ocupar aquellas reservas que se van a cancelar). El hotel está en el centro de Lisboa y en la misma ciudad el grupo hotelero tiene varios hoteles, por lo que **no hay ningún riesgo en llenar el hotel más de la cuenta** y enviar el exceso de reservas a otros hoteles en la ciudad.

## Descripción de la tarea

El dataset (**hotusa_cancellations.csv**) con el que trabajar contiene las siguientes columnas:
- *HotelId*: Id del hotel
- *ReservationStatusDate*: Fecha en la que se realizó la reserva
- *ArrivalDate*: Fecha de inicio de la reserva
- *LeadTime*: Antelación (días entre la generación de la reserva hasta la fecha de inicio de ésta)
- *StaysInWeekendNights*: Número de noches entre semana
- *StaysInWeekNights*: Número de noches el fin de semana
- *Adults*: Número de adultos en la reserva
- *Children*: Número de niños en la reserva
- *CustomerType*: Tipo de cliente
- *ADR*: Average Daily Rate (precio medio por reserva. Precio total de la reserva dividido por el número de noches)
- *Meal*: Tipo de Alojamiento (BB (Bed and Breakfast), HB (Half Board), Undefined, FB (Full Board), and SC (Self Catering))
- *Country*: País de procedencia del cliente
- *Company*: Compañía del cliente
- *ReservedRoomType*: Habitación reservada
- *IsRepeatedGuest*: Cliente repetitivo
- *IsCanceled*: Booleano que indica si la reserva se canceló o no (TARGET variable)

Este documento contiene los siguientes puntos dentro del Jupiter Notebook (`Grupo Hotusa_Data Scientist test_Soraya Alvarez_final.ipynb`):
1. **EDA sencillo y básico** (A Revenue Management le preocupan más los resultados que los insights)

Observaciones preliminares del análisis exploratorio de datos (EDA):

- Tipo de cliente: Mayor tasa de cancelación en "Transient" (16.2%) y "Transient-party" (9.3%).
- Tipo de alojamiento: Mayor tasa de cancelación en FB (Full Board: 36.8%) y HB (Half Board: 16.7%). Sin embargo, más cancelaciones totales en BB (3343) y HB (1102).
- País de origen: Mayor número de cancelaciones de Portugal (67%), seguido de Reino Unido (7.7%), España (7.7%) e Irlanda (4%). Mayor porcentaje de cancelaciones de Georgia, Moldavia, ARE y Pakistán (50%).
- Tipo de habitación: Mayor tasa de cancelación en "H" (26%), "G" (23%) y "L" (20%). Sin embargo, más cancelaciones totales en A (2743), D (800) y E (610).
- Huésped habitual: Menor tasa de cancelación para clientes frecuentes (3.3%) vs. nuevos clientes (14.7%).
- Lead Time (tiempo de antelación): Mayor correlación con la cancelación. Las reservas con mucha antelación tienen más probabilidades de cancelarse.
- Precio medio por habitación (ADR): Tendencia positiva con la cancelación, pero no hay diferencia estadísticamente significativa con las reservas no canceladas.
- Días de estancia: Aumenta la tasa de cancelación con el número de días, tanto en semana como en fin de semana. Sin embargo, más cancelaciones totales en estancias de 5 días entre semana, 2 días fin de semana, 2 adultos y sin niños.
- Variables temporales: Patrones dinámicos con la cancelación. Se recomienda analizar mes y año para identificar meses con mayor tasa de cancelación.
   
  * ¿Qué variables parecen ser las más predictivas?
    
  En base al EDA anterior, considero que la variable predictiva mas importante seria el tiempo de anticipacion de la reserva (LeadTime), seguida de las variables categoricas y temporales. Sin embargo, este resumen se basa en los insights de la EDA y puede estar incompleto o no reflejar todos los aspectos del análisis. Se recomienda revisar el EDA completo para obtener una comprensión más detallada de los hallazgos y analizar la importancia de las variables en el modelo.

2. **Preparación de los datos**
  * ¿Le darías el mismo tratamiento a todas las variables?
    No le daría el mismo tratamiento a todas las variables. Las variables categóricas necesitan ser transformadas en variables ficticias (dummies), y algunas de las variables numéricas podrían tratarse como variables categóricas, por lo que también tendríamos que convertirlas en variables dummies. Las variables numéricas deberíamos normalizarlas utilizando métodos como el logaritmo neperiano u otros métodos similares, para mejorar el rendimiento del modelo. Por ultimo, las variables con fechas, no podemos utilizarlas directamente, sino que las transformaremos en características (features) numéricas que el modelo pueda entender.
  
  * ¿Utilizarías todas las variables o eliminarías alguna de ellas por no ofrecer valor?
  Para mejorar el modelo y evitar problemas como la multicolinealidad, se propone eliminar las variables Unnamed: 0 (código de cliente), HotelId (innecesario al haber un solo hotel), StaysInWeekNights (se agrupa con StaysInWeekendNights en total_stays), Adults y Children (se agrupan en Total_guests), Company (clasificada como 0 o 1 según si la reserva la hizo una empresa). Las variables temporales ReservationStatusDate y ArrivalDate se transforman en dos variables cada una, representando el mes y el año de la fecha. Country se agrupa en nacionales (portugueses) e internacionales. Se conservan las demás variables para mantener un modelo lo más completo posible. La decisión final sobre la eliminación de variables puede revisarse en el Jupiter Notebook, y después de evaluar la precisión y otras métricas del modelo.

3. **Construcción del modelo**
  * ¿Qué tipo de modelo es el más adiente? Support Vector Machines, Decision Tree, Logistic Regression, K-Means, Redes Neuronales, Random Forest, Gradient Boosting, Naive Bayes, PCA, Lasso Regression...
  Se seleccionó XGBoost por su robustez a clases desbalanceadas (cancelaciones representan el 14%), capacidad para manejar variables mixtas y proporcionar interpretabilidad.
  
  * No es necesario probar entre cientos de modelos con distintos hiperparámetros para saber cuál da mejores resultados, basta con justificar la elección.
  
4. **Evaluación del modelo**
  * ¿Cuál sería el benchmark a batir? ¿Estamos contentos con los resultados obtenidos?
  El benchmark a batir es un modelo con una precisión y una AUC ROC superiores a 0.5 indican que el modelo es mejor que el azar. En nuestro caso obtuvimos una accuracy del 97% en los datos de prueba y una ROC AUC de 0.89. Aunque hay una leve tendencia al overfitting, esta discrepancia es aceptable y no representa una preocupación significativa. Es importante seguir monitoreando el modelo para detectar posibles problemas de overfitting a medida que se ajusta y se aplican técnicas adicionales de validación. Por ello, si estoy contenta con los resultados obtenidos.
    
  * ¿Qué resultados podríamos esperar con este modelo en producción?
    Los resultados que esperamos obtener con este modelo en produccion son la reduccion de cancelaciones efectivas al realizar overbooking de manera precisa. Este modelo clasifica si una reserva se cancelará o no con una precisión de hasta el 97%. Como resultado permite a los hoteles estimar sus tasas de ocupación de forma más cuantificable, gestionar su negocio adecuadamente y aumentar aún más sus ingresos.

    - Los riesgos de este modelo son los siguientes (basado en los datos de confusion matrix):
        - 3% de posibilidades de que el hotel prediga que no se cancela una reserva, y finalmente la reserva saldrá mal (reserva cancelada).
        - 2% de posibilidades de que el hotel no prediga/sospeche que la reserva se cancelara, y finalmente no se cancela, con el riesgo de overbooking. En estos casos, este % de reservas sera facilmente asumible por otros hoteles del mismo grupo en la ciudad de Lisboa.
