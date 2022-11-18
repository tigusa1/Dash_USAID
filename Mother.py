import numpy as np


class Mother:
    def __init__(self, wealth, education, age, no_children, max_gest_age, B):
        """initiate characteristics of Mother agent"""
        Health_const_0 = B['Health_const_0']
        Health_slope_0 = B['Health_slope_0']
        Predisp_ANC_const_0 = B['Predisp_ANC_const_0']
        Predisp_ANC_slope_0 = B['Predisp_ANC_slope_0']
        
        self._wealth = wealth
        self._education = education
        self._age = age
        self._no_children = no_children
        self.B = B

        self._gest_age = -(np.int(np.random.randint(-8, max_gest_age, 1)))
        self.logit_health = Health_const_0 + Health_slope_0 * (np.random.uniform(-1, 1, 1))
        self._health = logistic(self.logit_health)
        self._predisp_ANC = Predisp_ANC_const_0 + Predisp_ANC_slope_0 * (np.random.uniform(-1, 1, 1))


        self._delivery = None
        self._facility = None
        self._anc = 0
        self.delivered = False # needed to count the number of deliveries per month

    def visit_anc(self):
        """go to ANC if predisposition for it, changes health"""
        anc_treatment = 0.1
        if self._predisp_ANC > np.random.uniform(0, 1, 1):
            self.logit_health += 0.2*anc_treatment
            self._anc += 1

    def choose_delivery(self, l4_quality, l2_quality, proximity, health_outcomes, L2_net_capacity, L4_net_capacity):
        """delivery facility depending on where one goes for care and health status"""
        Health_outcomes__Predisp = self.B['Health_outcomes__Predisp']
        L4_Q__Predisp            = self.B['L4_Q__Predisp']
        Health_Predisp           = self.B['Health_Predisp']
        logit_predisp_l4 = 0.02*self._wealth + 0.02*self._education + 0.001*self._age \
                         + 0.05*self._no_children + Health_Predisp*self._health \
                         + L4_Q__Predisp*l4_quality + 0.1*proximity \
                         + Health_outcomes__Predisp*health_outcomes
        logit_predisp_l2_l4 = logit_predisp_l4 + 1
        rand = np.random.uniform(0, 1, 1)
        if logistic([logit_predisp_l4 - 2]) > rand:
            self._facility = 2
            self.logit_health += 6*(l4_quality-1/2) + 1.5
        elif logistic([logit_predisp_l2_l4 - 2]) > rand:
            if L2_net_capacity > 0.0: # if there is room
                self._facility = 1
                self.logit_health += 6*(l2_quality-1/2) + 0.5
            else: # otherwise, go to level 4/5, but not as healthy
                self._facility = 2
                self.logit_health += 6*(l4_quality-1/2) + 1.5 - 0.5
        else:
            self._facility = 0
            self.logit_health += -0.5

    def deliver(self):
        """delivery outcome"""
        self.delivered = True
        if logistic([self.logit_health]) < np.random.uniform(0, 1, 1):
            self._delivery = -1
        else:
            self._delivery = 1

        if logistic([self.logit_health,1]) < np.random.uniform(0, 1, 1):
            self.baby_delivery = -1
        else:
            self.baby_delivery = 1

    def increase_age(self, l4_quality, l2_quality, proximity, health_outcomes, L2_net_capacity, L4_net_capacity):
        """increase gestational age (step)"""
        self._gest_age = self._gest_age + 1
        if (self._gest_age > 0) & (self._gest_age < 9):
            self.visit_anc()
        elif self._gest_age == 9:
            self.choose_delivery(l4_quality, l2_quality, proximity, health_outcomes, L2_net_capacity, L4_net_capacity)
            self.deliver()
        self._health = logistic(self.logit_health)


def logistic(x):
   return 1 / (1 + np.exp(-np.mean(x)))