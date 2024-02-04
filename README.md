# Symmetric and Asymmetic Development of Bluetooth-Based Indoor Localization Mechanisms

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/oeg-upm/TINTOlib-Documentation/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)


## Citing these works

**Citing Asymmetric withouth optimization**: If you used assymetric code in your work, please cite the **[Sensors](https://www.mdpi.com/1424-8220/17/6/1318)**:

```bib
@Article{s17061318,
    AUTHOR = {Castillo-Cara, Manuel and Lovón-Melgarejo, Jesús and Bravo-Rocca, Gusseppe and Orozco-Barbosa, Luis and García-Varea, Ismael},
    TITLE = {An Empirical Study of the Transmission Power Setting for Bluetooth-Based Indoor Localization Mechanisms},
    JOURNAL = {Sensors},
    VOLUME = {17},
    YEAR = {2017},
    NUMBER = {6},
    ARTICLE-NUMBER = {1318},
    PubMedID = {28590413},
    ISSN = {1424-8220}
    DOI = {10.3390/s17061318}
}
```

And Assymetric with optimization code: **[IEEE Access](https://ieeexplore.ieee.org/abstract/document/8642816)** 

```bib
@ARTICLE{8642816,
    author={Lovón-Melgarejo, Jesús and Castillo-Cara, Manuel and Huarcaya-Canal, Oscar and Orozco-Barbosa, Luis and García-Varea, Ismael},
    journal={IEEE Access}, 
    title={Comparative Study of Supervised Learning and Metaheuristic Algorithms for the Development of Bluetooth-Based Indoor Localization Mechanisms}, 
    year={2019},
    volume={7},
    number={},
    pages={26123-26135},
    doi={10.1109/ACCESS.2019.2899736}
  }
```

And if you use the smartphone data: **[Jounal of Sensors](https://www.hindawi.com/journals/js/2017/1928578/)** 

```bib
@ARTICLE{1928578,
    author={Castillo-Cara, Manuel and Lovón-Melgarejo, Jesús and Bravo-Rocca, Gusseppe and Orozco-Barbosa, Luis and García-Varea, Ismael},
    journal={Journal of Sensors}, 
    title={An analysis of multiple criteria and setups for Bluetooth smartphone-based indoor localization mechanism}, 
    year={2017},
    volume={2017},
    number={1928578},
    pages={22},
    doi={10.1155/2017/1928578}
  }
```


## Abstract

The present papers constitute a seminal contribution to indoor localization utilizing Bluetooth signals and supervised learning algorithms. They elucidate pivotal scientific advancements that solidify research endeavors in this domain. The manuscripts scrutinize several pivotal facets:

- Bluetooth signal characterization: The investigation delves into the behavioral patterns of Bluetooth signals employing 12 distinct supervised learning algorithms. This characterization serves as a cornerstone for comprehending the intricacies of signal dynamics, laying the groundwork for fingerprinting-based localization mechanisms.
- Optimization of transmit power configuration: The utilization of metaheuristic algorithms for optimizing the configuration of transmit powers is extensively discussed. This endeavor is paramount for enhancing the efficacy of indoor localization systems by mitigating the computational complexities associated with brute-force search methodologies.
- Assessment of symmetrical and asymmetrical transmission power configurations: The paper thoroughly evaluates transmission power configurations aimed at mitigating the impacts of multipath fading. The analysis juxtaposes the performance of diverse classification models under varied configurations, elucidating the settings that yield optimal outcomes. These contributions epitomize significant strides towards the continual enhancement of indoor localization systems, bearing profound implications for their operational efficiency and reliability."


## Getting Started
The project has the following folders
- **Data**: Includes data obtained by the mobile phone and also by the Raspberry Pi.
- **Asymmetric Code**: Includes Python code of the results with different transmission powers using evolutionary and genetic algorithms.
- **Asymmetric Code**: It includes results obtained with classical machine learning algorithms with asymmetric transmission power. It is used for smartphone data and also for Raspberri Pi data.git

## License

TINTOlib is available under the **[Apache License 2.0](https://github.com/oeg-upm/TINTOlib-Documentation/blob/main/LICENSE)**.

## Authors
- **[Manuel Castillo-Cara](https://github.com/manwestc)**


## Institutions

<kbd><img src="https://www.uni.edu.pe/images/logos/logo_uni_2016.png" alt="Universidad Nacional de Ingeniería" width="110"></kbd>
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-oeg.png" alt="Ontology Engineering Group" width="100"></kbd> 
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-upm.png" alt="Universidad Politécnica de Madrid" width="100"></kbd>
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="80"></kbd> 
