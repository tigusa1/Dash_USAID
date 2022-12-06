import numpy as np
import random

class Mother_simplified:
    def __init__(self, max_gest_age, B, unique_id, df, Health_const_0, Health_slope_0):
    # def __init__(self, max_gest_age, Health_const_0, Health_slope_0):
        """initiate characteristics of Mother agent"""
        self.logit_health = Health_const_0 + Health_slope_0 * (np.random.uniform(-1, 1, 1))
        self.logit_health_BL = self.logit_health
        self._health = logistic(self.logit_health)

        self._gest_age = -(np.int(np.random.randint(-9, max_gest_age, 1)))
        self._delivery = None
        self._facility = None
        self.delivered = False  # needed to count the number of deliveries per month

        self.flag_network = False

        if self.flag_network:
            self._network = []

            self._id = unique_id
            self._SES = df['wealth'][unique_id]
            self._location = df['new_lat_long'][unique_id]

            predisp = np.int(np.random.randint(0,3,1)) # a random integer from 0 to 2
            self._l4 = [int(predisp == 2)]
            self._l2 = [int(predisp == 1)]
            self._home = [int(predisp == 0)]

            self._L4_Q_Predisp = np.sum(self._l4)
            self._Predisp_L2_L4 = np.sum(self._l2)
            self._time_CHV = int(np.random.randint(0, 8, 1))
            self._CHV = 0

        self.B = B

    def set_B(self, B):
        pass

    def build_Net(self, mothers):
        for idx, mother in enumerate(mothers):
            if (self._id != idx) and (np.linalg.norm(np.array(self._location) - np.array(mother._location)) < 0.05) and (self._SES == mother._SES):
                self._network.append(idx) 

    def influence_Net(self, mothers):
        friends = int(len(self._network)/2)
        net = random.sample(self._network, friends)
        for mother in net:
            mothers[mother]._l4.append(int((self._facility == 2) and (self._delivery == 1)))
            mothers[mother]._l2.append(int((self._facility == 1) and (self._delivery == 1)))
            mothers[mother]._l2.append(int((self._facility == 0) and (self._delivery == 1)))

    def update_predisp(self):
        CHV_weight = 0.3
        Rec_weight = 0.2
        if self._l4[-1] == 2:
            L4_predisp = (1-CHV_weight)*self._l4[:-1] + CHV_weight*self._l4[-1]
        else:
            L4_predisp = (1-Rec_weight)*np.sum(self._l4[:-1])/len(self._l4) + Rec_weight*self._l4[-1]/len(self._l4)
            L2_predisp = (1-Rec_weight)*np.sum(self._l2[:-1])/len(self._l2) + Rec_weight*self._l2[-1]/len(self._l2)
            home = (1-Rec_weight)*np.sum(self._home[:-1])/len(self._home) + Rec_weight*self._home[-1]/len(self._home)

        self._L4_Q_Predisp = L4_predisp/(L4_predisp + L2_predisp + home)
        self._Predisp_L2_L4 = L2_predisp/(L4_predisp + L2_predisp + home)

    def see_CHV(self):
        if (np.random.binomial(1, 0.2, 1) == 1) and (self._CHV == 0):
            self._CHV == 1
            self._l4.append(1)

    def choose_delivery(self, l4_quality, l2_quality, health_outcomes, L2_net_capacity):
        """delivery facility depending on where one goes for care and health status"""

        if self.flag_network:
            self.B['L4_Q__Predisp'] = self._L4_Q_Predisp
            self.B['Predisp_L2_L4'] = self._Predisp_L2_L4

        prob_l4, prob_l2, logit_health_l4, logit_health_l2, logit_health_l4_l2, logit_health_l0 = \
            get_prob_logit_health(self.B, l4_quality, l2_quality, health_outcomes, self.logit_health)

        rand = np.random.uniform(0, 1, 1)
        if prob_l4 > rand:
            self._facility = 2
            self.logit_health = logit_health_l4 # average initial self.logit_health is Health_const_0
        else:
            rand = np.random.uniform(0, 1, 1)
            if prob_l2 > rand:
                if L2_net_capacity > 0.0:  # if there is room
                    self._facility = 1
                    self.logit_health = logit_health_l2
                else:  # otherwise, go to level 4/5, but not as healthy
                    self._facility = 2
                    self.logit_health = logit_health_l4_l2
            else:
                self._facility = 0
                self.logit_health = logit_health_l0

    def deliver(self):
        """delivery outcome"""
        self.delivered = True
        if logistic([self.logit_health]) < np.random.uniform(0, 1, 1):
            self._delivery = -1
        else:
            self._delivery = 1

        if logistic([self.logit_health, 1]) < np.random.uniform(0, 1, 1):
            self.baby_delivery = -1
        else:
            self.baby_delivery = 1

    def increase_age(self, l4_quality, l2_quality, health_outcomes, L2_net_capacity, mothers, ts):
        """increase gestational age (step)"""
        self._gest_age = self._gest_age + 1

        if self.flag_network:
            if ts == 0:
                self.build_Net(mothers)

            if self._gest_age == self._time_CHV:
                self.see_CHV()

            self.update_predisp()

        if self._gest_age == 9:
            self.choose_delivery(l4_quality, l2_quality, health_outcomes, L2_net_capacity)
            self.deliver()
            if self.flag_network:
                self.influence_Net(mothers)

        self._health = logistic(self.logit_health)

def get_prob_logit_health(B, l4_quality, l2_quality, health_outcomes, logit_initial):
    Health_const_0 = B['Health_const_0']
    Health_slope_0 = B['Health_slope_0']

    Health_outcomes__Predisp = B['Health_outcomes__Predisp']
    L4_Q__Predisp = B['L4_Q__Predisp']

    Predisp_L2_L4 = B['Predisp_L2_L4']  # 1
    Q_Health_multiplier = B['Q_Health_multiplier']  # 6
    Q_Health_L4_constant = B['Q_Health_L4_constant']  # 1.5
    Q_Health_L4_L2_difference = B['Q_Health_L4_L2_difference']  # 1
    Q_Health_L4_referral_difference = B['Q_Health_L4_referral_difference']  # 0.5
    Q_Health_Home_negative = B['Q_Health_Home_negative']  # 0.5

    Initial_Negative_Predisp = B['Initial_Negative_Predisp']  # 0

    logit_predisp_l4 = L4_Q__Predisp * l4_quality \
                       + Health_outcomes__Predisp * health_outcomes \
                       - Initial_Negative_Predisp
    logit_predisp_l2_l4 = logit_predisp_l4 + Predisp_L2_L4

    prob_l4 = logistic([logit_predisp_l4 - 2])
    prob_l2 = logistic([logit_predisp_l2_l4 - 2])

    # logit_health_BL    = Health_const_0 + Health_slope_0 * (np.random.uniform(-1, 1, 1)) # not used for initialization

    logit_health_l4    = logit_initial + Q_Health_multiplier * (l4_quality - 1 / 2) + Q_Health_L4_constant
    logit_health_l2    = logit_initial + Q_Health_multiplier * (l2_quality - 1 / 2) + \
                                     Q_Health_L4_constant - Q_Health_L4_L2_difference
    logit_health_l4_l2 = logit_initial + Q_Health_multiplier * (l4_quality - 1 / 2) + \
                                     Q_Health_L4_constant - Q_Health_L4_referral_difference
    logit_health_l0    = logit_initial - Q_Health_Home_negative

    return prob_l4, prob_l2, logit_health_l4, logit_health_l2, logit_health_l4_l2, logit_health_l0

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
        self.logit_health_BL = self.logit_health
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

        Predisp_L2_L4  = self.B['Predisp_L2_L4']  # 1
        Q_Health_multiplier  = self.B['Q_Health_multiplier']  # 6
        Q_Health_L4_constant  = self.B['Q_Health_L4_constant']  # 1.5
        Q_Health_L4_L2_difference  = self.B['Q_Health_L4_L2_difference']  # 1
        Q_Health_L4_referral_difference  = self.B['Q_Health_L4_referral_difference']  # 0.5
        Q_Health_Home_negative  = self.B['Q_Health_Home_negative']  # 0.5

        Wealth__Predisp = self.B['Wealth__Predisp']  # 0.2
        Education__Predisp = self.B['Education__Predisp']  # 0.02
        Age__Predisp = self.B['Age__Predisp']  # 0.001
        No_Children__Predisp = self.B['No_Children__Predisp']  # 0.05
        Proximity__Predisp = self.B['Proximity__Predisp']  # 0.1
        Initial_Negative_Predisp = self.B['Initial_Negative_Predisp'] # 0

        logit_predisp_l4 = Wealth__Predisp*self._wealth + Education__Predisp*self._education\
                         + Age__Predisp*self._age \
                         + No_Children__Predisp*self._no_children + Health_Predisp*self._health \
                         + L4_Q__Predisp*l4_quality + Proximity__Predisp*proximity \
                         + Health_outcomes__Predisp*health_outcomes \
                         - Initial_Negative_Predisp
        logit_predisp_l2_l4 = logit_predisp_l4 + Predisp_L2_L4
        rand = np.random.uniform(0, 1, 1)
        if logistic([logit_predisp_l4 - 2]) > rand:
            self._facility = 2
            self.logit_health += Q_Health_multiplier*(l4_quality-1/2) + Q_Health_L4_constant
        else:
            rand = np.random.uniform(0, 1, 1)
            if logistic([logit_predisp_l2_l4 - 2]) > rand:
                if L2_net_capacity > 0.0: # if there is room
                    self._facility = 1
                    self.logit_health += Q_Health_multiplier*(l2_quality-1/2) + \
                                         Q_Health_L4_constant - Q_Health_L4_L2_difference
                else: # otherwise, go to level 4/5, but not as healthy
                    self._facility = 2
                    self.logit_health += Q_Health_multiplier*(l4_quality-1/2) + \
                                         Q_Health_L4_constant - Q_Health_L4_referral_difference
            else:
                self._facility = 0
                self.logit_health += -Q_Health_Home_negative

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