
import random
from collections import Counter
from enum import Enum
from geopy.distance import great_circle
from geopy import Point
import geohash
import numpy as np




"""
    ------------------------------------------------------
    --  Discrete Event Counting & Reporting:
    ------------------------------------------------------
"""

class DO:
    def __init__(self, kwargs:dict):
        for k,v in kwargs.items():
            setattr(self, '_'+str(k),v)
    def __str__(self):
        n = self.__class__.__name__
        return str(type(self))+'\n'+('\n'.join([f' - {k}:\t{v}' 
                                                for k,v in self.__dict__.items() 
                                                if not k.startswith(f'_{n}__') ]))

class IncCounter(Counter):
    def inc(self, k): self[k] += 1
    def __str__(self):
        return str(type(self))+'\n'+('\n'.join([f'  - {k}:\t{v}' for k,v in self.items()]))
    def to_dict(self, name:str=None, prefix:str=''):
        d = {prefix+k:v for k,v in self.items()}
        if name:
            d['name'] = name
        return d

print_once = {}
class WARNING:
    @staticmethod
    def print_once(k,t):
        global print_once
        if k not in print_once:
            print_once[k] = True
            print(t)
          




"""
    ------------------------------------------------------
    --  Hospital Model:
    ------------------------------------------------------
"""

class Patient: pass # Required to specify mutual-referencing.

CareLevel = Enum('CareLevel', ['PC','SC','TC'])

class HospitalLookup(DO):
    def __init__(self, dfh:pd.DataFrame, precision:int=4, kwargs:dict={}):
        super().__init__(kwargs)
        self.__dfh = dfh
        self.precision = precision
    def distance_km(self, lat:float, lng:float, g:str) -> float:
        return great_circle( Point(lat,lng), Point(geohash.decode(g)) ).meters/1000
    def closest_PC(self, lat:float, lng:float):
        return self._get_nearby_hospitals(lat, lng, CareLevel.PC)
    def closest_hospital(self, lat:float, lng:float, careLevel:Enum):
        return self._get_nearby_hospitals(lat, lng, careLevel)
    def _index(self, lat:float, lng:float, idx:int):
        _df = pd.DataFrame(self.__dfh.loc[idx].copy()).T
        _df['km'] = _df['geohash'].apply( lambda x: self.distance_km(lat, lng, x) )
        return _df
    def _bayes_selection(self,probs):
        def roulette(probs:np.array):
            prev = 0.0
            roulette = [[probs[0][0], prev]]
            for i in range(len(probs)):
                curr = prev+probs[i][1]
                roulette += [[probs[i][0], curr]]
                prev = curr
            return roulette
        def select_index(roulette, v:float):
            c_id = roulette[0][0]
            prev = roulette[0][1]
            for idx,p in roulette[1:]:
                if prev < v <= p:
                    c_id = idx
                    break
            return int(c_id), v, roulette
        return select_index(roulette(probs), random.random())
    
    def _get_nearby_hospital(self, patient:Patient, lat:float, lng:float, careLevel:Enum):
        lookup = { CareLevel.TC: ['Psychiatric'],
                   CareLevel.SC: ['A','S','S or M1','M1'],
                   CareLevel.PC: ['M2 or smaller','PrimaryCare']
                 }
        # print('![TODO] expand search area if no hospitals available.')
        # print('![TODO] measure distances to hospitals')
        # print('![TODO] apply bayesian probs to hospitals')
        # print('![TODO] Add all Psychiatric hospitals.')
        WARNING.print_once(k='TC_REFERRAL', t="""
                    ![TODO] Add all Psychiatric hospitals.
                    """)
        for precision in range(self.precision,0,-1):
            try:
                _geohash = geohash.encode(lat, lng, precision)
            except ValueError as e:
                print(e)
                print(f'lat:{lat}, lng:{lng}, precision:{precision}')
                print(patient)
                raise e
            queries = [f'`geohash`.str.startswith(\'{x}\')' for x in geohash.expand(_geohash)]
            _df = self.__dfh[self.__dfh['Category'].isin(lookup[careLevel])].query(' | '.join(queries))
            if len(_df)>0:
                break
        _df = _df.copy()
        _df['km'] = _df['geohash'].apply( lambda x: self.distance_km(lat, lng, x) )
        _df = _df.sort_values(by='km').head(5)
        _df['bayes'] = 1/_df['km']
        _df['bayes'] = _df['bayes'] / _df['bayes'].sum()
        probs = _df.reset_index()[['id','bayes']].to_numpy()
        idx, _, _ = self._bayes_selection(probs)
        return pd.DataFrame(_df.loc[idx]).T


"""
    -------------------------------------------------------------------
    --  Optimization Function for Large Patient/Hospital Datasets:
    -------------------------------------------------------------------
"""
def apply_nearby_hospitals_to_df(dfh, df_study):
    """
    ##### Apply Nearby Hospitals (PC, SC, TC) to each Patient Record:
    Input:
      @dfh [pandas.dataframe] of hospitals.
      @df_study [pandas.dataframe] of patients.
    Effects:
      @df_study is modified.
    
    - Optimization for large datasets. 
    - Requests a nearby hospital (PC/SC/TC care levels) and stores its geohash into a dataframe.
    """
    hospitals = HospitalLookup(dfh, precision=4)
    for k in CareLevel:
        c = f'nearby_{k.name.lower()}_id'
        if c not in df_study.columns:
            df_study[c] = float('nan')
        
    for row in tqdm(df_study[~df_study['latitude'].isna()][['latitude','longitude']].iterrows()):
        for k in CareLevel:
            r = hospitals._get_nearby_hospital(None, row[1]['latitude'], row[1]['longitude'], k)
            df_study.loc[row[0],f'nearby_{k.name.lower()}_id'] = r.index[0]

"""
    ------------------------------------------------------
    --  Measurable Events Declaration:
    ------------------------------------------------------
"""

Event = Enum('Event', ['TRAVEL',
                       'VISIT_PC',
                       'VISIT_SC',
                       'VISIT_TC',
                       'VISIT_HV',
                       'TRAVEL_HV',
                       'REFER_TO_PC',
                       'REFER_TO_SC',
                       'REFER_TO_TC',
                       'TEST_2Q',
                       'TEST_9Q',
                       'TEST_8Q',
                       'TEST_RO_PHYSICAL',
                       'TEST_RO_MEDICINE',
                       'TEST_MDD_DIAGNOSIS',
                       'TEST_RO_PSYCHOSOCIAL',
                       'NOTIFY_REPORT',
                       'TREATMENT_ED',
                       'TREATMENT_CSG',
                       'TREATMENT_Rx',
                       'FOLLOWUP_HV_9Q',
                       'FOLLOWUP_PC_9Q',
                       'FOLLOWUP_SC_9Q_Rx',
                      ])
class Events(IncCounter): pass
class Travel(IncCounter): pass

"""
    ------------------------------------------------------
    --  Patient Model:
    ------------------------------------------------------
"""


class Patient(DO):
    def __init__(self, config, events:Events, travel:Travel, hosp:HospitalLookup=None, kwargs:dict={}):
        super().__init__(kwargs)
        self.__events = events
        self.__travel = travel
        self.__hosp = hosp
        self._MDD_PERC = config._mdd #0.694
        self._is_PsySocial = self.__rand_perc(config._psy)
        self._is_SDDPReferToPC_vs_SC = self.__rand_perc(config._rpsc)
        self._is_FollowupPC_vs_HV = self.__rand_perc(config._fhv)
        self._is_VisitPC_vs_HV = self._is_FollowupPC_vs_HV
        self._is_EarlyFollowUpDropout_PC = self.__rand_perc(config._fpc)
        self._is_EarlyFollowUpDropout_SC = self.__rand_perc(config._fsc)
        if config._uacc>0 and config._uacch>0:
            raise Exception('Paramters `uacc` and `uacch` are both defined (>0); use only 1 at a time.')
        elif config._uacc>0:
            self._9q = self._rating_moderate_uncertainty_9q( config._uacc )
        elif config._uacch>0:
            self._9q = self._rating_coarse_uncertainty_9q( config._uacch )
            # self._9q = self._rating_fine_uncertainty_9q( config._uacch )
        self._selected_hospital = {k:None for k in CareLevel._member_names_}
        self._is_MDD = None
    def event(self, k):
        if k.name.startswith('TRAVEL'):
            raise ValueError(f'Deprecated: Logging `TRAVEL` events via Patient.event() are no longer allowed.\n Use Patient.travel() to log KM distances and events.')
        self.__events.inc(k.name)
    def travel(self, _event:Enum, careLevel:Enum):
        self.__events.inc(_event.name)
        self.__travel[_event.name] += self.bayesian_hospital_km(careLevel)
    def is_MDD(self):
        if self._is_MDD is None:
            MDD_percentage_of_9Q_GTEQ_7 = self._MDD_PERC
            is_mdd = False
            if self._9q >= 7:
                is_mdd = random.random() <= MDD_percentage_of_9Q_GTEQ_7 
                # 0 to 0.694 is MDD. 0.694 to 1.0 is not MDD.
            self._is_MDD = is_mdd
        return self._is_MDD
    def __rand_perc(self,p:float):
        # return random.randint(0,1)
        # return random.uniform(0,1) >= p
        return random.random() <= p # Fastest. Empirically Approx-Uniform.
    def __rand_50perc(self):
        return self.__rand_perc(0.5)
    def is_SDDPReferToPC_vs_SC(self):
        return self._is_SDDPReferToPC_vs_SC
    def is_VisitPC_vs_HV(self):
        return self._is_VisitPC_vs_HV
    def is_FollowupPC_vs_HV(self):
        return self._is_FollowupPC_vs_HV
    def is_PsySocial(self):
        return self._is_PsySocial
    def is_EarlyFollowUpDropout_PC(self):
        return self._is_EarlyFollowUpDropout_PC
    def is_EarlyFollowUpDropout_SC(self):
        return self._is_EarlyFollowUpDropout_SC
    def bayesian_hospital(self, careLevel:Enum):
        if self._selected_hospital[careLevel.name] is None:
            hosp_id = getattr(self, f'_nearby_{careLevel.name.lower()}_id')
            self._selected_hospital[careLevel.name] = self.__hosp._index(self._latitude, self._longitude, hosp_id)
        return self._selected_hospital[careLevel.name]
    def __bayesian_hospital_lookup(self, careLevel:Enum):
        if self._selected_hospital[careLevel.name] is None:
            self._selected_hospital[careLevel.name] = self.__hosp._get_nearby_hospital(self, self._latitude, self._longitude, careLevel)
        return self._selected_hospital[careLevel.name]
    def bayesian_hospital_km(self, careLevel:Enum):
        return self.bayesian_hospital(careLevel)['km'].iloc[0]
    def _rating_moderate_uncertainty_9q(self, uacc:float):
        global UncertaintyAccuracy_9Q_Moderate
        r = self._9q
        if self.__rand_perc(uacc): #is_9q_uncertain_accuracy -> Make a change or not.
            r = UncertaintyAccuracy_9Q_Moderate.response( self._9q ) or self._9q  #inc, dec or no-change rating (retained as score value)
        return r
    def _rating_fine_uncertainty_9q(self, uacc:float):
            global UncertaintyAccuracy_9Q_Fine_Inc,UncertaintyAccuracy_9Q_Fine_Dec
            r = self._9q
            if self.__rand_perc(uacc): #is_9q_uncertain_accuracy -> Make a change or not.
                if self.__rand_perc(.5):
                    r = UncertaintyAccuracy_9Q_Fine_Inc.response( self._9q ) or self._9q  #inc, dec or no-change rating (retained as score value)
                else:
                    r = UncertaintyAccuracy_9Q_Fine_Dec.response( self._9q ) or self._9q  #inc, dec or no-change rating (retained as score value)
            return r
    def _rating_coarse_uncertainty_9q(self, uacc:float):
            global UncertaintyAccuracy_9Q_Coarse_Inc,UncertaintyAccuracy_9Q_Coarse_Dec
            r = self._9q
            if self.__rand_perc(uacc): #is_9q_uncertain_accuracy -> Make a change or not.
                if self.__rand_perc(.5):
                    r = UncertaintyAccuracy_9Q_Coarse_Inc.response( self._9q ) or self._9q  #inc, dec or no-change rating (retained as score value)
                else:
                    r = UncertaintyAccuracy_9Q_Coarse_Dec.response( self._9q ) or self._9q  #inc, dec or no-change rating (retained as score value)
            return r


"""
    ------------------------------------------------------
    --  Discrete Models:
    ------------------------------------------------------
"""



class Proposed_HV_TestPsySoc:
    """
    ### Proposed_HV_TestPsySoc
    - This method applies Followups as HV->P (zero follow-ups are as P->PC).
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [Our Proposed_HV_TestPsySoc] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TEST_9Q)
            
                is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
                if is_viable_valid_9Q:
                    if p._9q > 0 and p._9q < 7: # Unspecified
                        p.event(Event.TEST_RO_PSYCHOSOCIAL)
                        if p.is_PsySocial():
                            p.event(Event.TREATMENT_ED)
                            p.event(Event.REFER_TO_SC)
                            p.travel(Event.TRAVEL, CareLevel.SC)
                            Proposed_HV_TestPsySoc.SC.receive_referral_PsySoc_CSG(p)
                        else:
                            p.event(Event.TREATMENT_ED)
                        Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                    elif p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_HV_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_HV_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 19: # Severe
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        WARNING.print_once(k='HV_REFER_TC', t="""
                            [!NOTE] HV could refer directly to SC or TC... 
                                    Well, yes. Because "Refer" is to "recommend" and not "Diagnose" or "Admit".
                                    This would save Severe-P from extra TRAVEL/VISIT and TEST events.
                                    However, there aren't many Severe-P. 
                                    "Wait time to care" KPI would improve; but distance KPI would not improve.
                                    More training is required ... Well not really, as 9Q is easy to interpret.
                                    Could be a wasteful "overshoot" if non-MDD circumstances are the cause.
                                    Probably best, in the end, to refer to SC for GP diagnosis.
                            """)
                        Proposed_HV_TestPsySoc.SC.receive_referral_MDD_Diag(p)
            else:
                pass
        @staticmethod
        def followup_9Q(p:Patient):
            p.event(Event.FOLLOWUP_HV_9Q)
            interval_months = 1
            duration_months = 12
            duration_months = duration_months if not p.is_EarlyFollowUpDropout() else duration_months//2
            for i in range(1, duration_months+1, interval_months):
                p.event(Event.TEST_9Q)
                p.event(Event.VISIT_HV)
                p.travel(Event.TRAVEL_HV, CareLevel.PC)


    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral_PsySoc_CSG(p:Patient):
            p.event(Event.TREATMENT_CSG)
            Proposed_HV_TestPsySoc.HV.followup_9Q(p)
        
        @staticmethod
        def receive_referral_MDD_Diag(p:Patient):
            p.event(Event.VISIT_SC)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                p.event(Event.TEST_8Q)
                p.event(Event.TEST_RO_PHYSICAL)
                p.event(Event.TEST_RO_MEDICINE)
                p.event(Event.TEST_MDD_DIAGNOSIS)
                if not p.is_MDD():
                    p.event(Event.TEST_RO_PSYCHOSOCIAL)
                    if p.is_PsySocial():
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_CSG)
                        Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                    else:
                        p.event(Event.TREATMENT_ED)
                else: # is MDD:
                    if p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        Kongsuk.SC.followup_9Q_Rx(p)

                    elif p._9q >= 19: # Severe
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        p.event(Event.REFER_TO_TC)
                        p.travel(Event.TRAVEL, CareLevel.TC)
                        Kongsuk.TC.receive_referral(p)






class Proposed_SC_TestPsySoc: 
    """
    ### Proposed_SC_TestPsySoc
    - This method applies Followups as HV->P (zero follow-ups are as P->PC).
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [Our Proposed_SC_TestPsySoc] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TEST_9Q)
            
                is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
                if is_viable_valid_9Q:
                    if p._9q > 0 and p._9q < 7: # Unspecified
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 19: # Severe
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
            else:
                pass


    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral_MDD_Diag(p:Patient):
            p.event(Event.VISIT_SC)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                p.event(Event.TEST_8Q)
                p.event(Event.TEST_RO_PHYSICAL)
                p.event(Event.TEST_RO_MEDICINE)
                p.event(Event.TEST_MDD_DIAGNOSIS)
                if not p.is_MDD():
                    p.event(Event.TEST_RO_PSYCHOSOCIAL)
                    if p.is_PsySocial():
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_CSG)
                    else:
                        p.event(Event.TREATMENT_ED)
                    Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                        
                else: # is MDD:
                    if p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                        
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        Kongsuk.SC.followup_9Q_Rx(p)

                    elif p._9q >= 19: # Severe
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        p.event(Event.REFER_TO_TC)
                        p.travel(Event.TRAVEL, CareLevel.TC)
                        Kongsuk.TC.receive_referral(p)
        




class Proposed_HV_NOP_9Q_Unspecified:
    """
    ### Proposed_HV_NOP_9Q_Unspecified
    - This method applies TREATMENT_ED for HV-9Q==Unspecified, identically to `Kongsuk's PC 9Q==Unspecfied -> TREATMENT_ED.`.
    - This method applies Followups as HV->P (zero follow-ups are as P->PC).
    - Otherwise, it follows the `Proposed_SC_TestPsySoc` method.
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [Our Proposed_SC_TestPsySoc] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TEST_9Q)
            
                is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
                if is_viable_valid_9Q:
                    if p._9q > 0 and p._9q < 7: # Unspecified
                        p.event(Event.TREATMENT_ED)
                    elif p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 19: # Severe
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_SC_TestPsySoc.SC.receive_referral_MDD_Diag(p)
            else:
                pass



class Proposed_HV_NOP_9Q_Unspecified_5050_PCHV_Followup:
    """
    ### Proposed_HV_NOP_9Q_Unspecified , 50:50 HV/PC Followups:
    - This method applies TREATMENT_ED for HV-9Q==Unspecified, identically to `Kongsuk's PC 9Q==Unspecfied -> TREATMENT_ED.`.
    - This method applies 50:50 rule for Followups as HV->P or P->PC.
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [Our Proposed_SC_TestPsySoc] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TEST_9Q)
            
                is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
                if is_viable_valid_9Q:
                    if p._9q > 0 and p._9q < 7: # Unspecified
                        p.event(Event.TREATMENT_ED)
                    elif p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_HV_NOP_9Q_Unspecified_5050_PCHV_Followup.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_HV_NOP_9Q_Unspecified_5050_PCHV_Followup.SC.receive_referral_MDD_Diag(p)
                    elif p._9q >= 19: # Severe
                        p.event(Event.REFER_TO_SC)
                        p.travel(Event.TRAVEL, CareLevel.SC)
                        Proposed_HV_NOP_9Q_Unspecified_5050_PCHV_Followup.SC.receive_referral_MDD_Diag(p)
            else:
                pass
    
    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral_MDD_Diag(p:Patient):
            p.event(Event.VISIT_SC)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                p.event(Event.TEST_8Q)
                p.event(Event.TEST_RO_PHYSICAL)
                p.event(Event.TEST_RO_MEDICINE)
                p.event(Event.TEST_MDD_DIAGNOSIS)
                if not p.is_MDD():
                    p.event(Event.TEST_RO_PSYCHOSOCIAL)
                    if p.is_PsySocial():
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_CSG)
                    else:
                        p.event(Event.TREATMENT_ED)
                    
                    if p.is_FollowupPC_vs_HV():
                        Kongsuk.PC.followup_9Q(p)
                    else:
                        Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                else: # is MDD:
                    if p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        if p.is_FollowupPC_vs_HV():
                            Kongsuk.PC.followup_9Q(p)
                        else:
                            Proposed_HV_TestPsySoc.HV.followup_9Q(p)
                        
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        Kongsuk.SC.followup_9Q_Rx(p)

                    elif p._9q >= 19: # Severe
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        p.event(Event.REFER_TO_TC)
                        p.travel(Event.TRAVEL, CareLevel.TC)
                        Kongsuk.TC.receive_referral(p)

class SDDP2013:
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [SDDP2013] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TREATMENT_ED)
                if p.is_SDDPReferToPC_vs_SC():
                    p.event(Event.REFER_TO_PC)
                    p.travel(Event.TRAVEL, CareLevel.PC)
                    SDDP2013.PC.receive_referral(p)
                else:
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    SDDP2013.SC.receive_referral(p)
            else:
                pass

    class PC(DO):
        """ PC - (Tests: 2Q, 9Q. Treatments: Ed, Report. Follow-up: 9Q. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_PC)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
            if is_viable_valid_9Q:
                if p._9q > 0 and p._9q < 7:
                    pass
                elif p._9q >= 7:
                    p.event(Event.TEST_8Q) # General Policy to exclude 8Q data results from analysis.
                    Kongsuk.PC.followup_9Q(p) # Note: This is in the flowchart diagram.
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    SDDP2013.SC.receive_referral(p)
                    

    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_SC)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                if p._9q >= 7:
                    p.event(Event.TEST_8Q)
                    p.event(Event.TEST_RO_PHYSICAL)
                    p.event(Event.TEST_RO_MEDICINE)
                    p.event(Event.TEST_MDD_DIAGNOSIS)
                    if not p.is_MDD():
                        p.event(Event.TEST_RO_PSYCHOSOCIAL)
                        if p.is_PsySocial():
                            p.event(Event.TREATMENT_CSG)
                            Kongsuk.PC.followup_9Q(p)
                        else:
                            p.event(Event.TREATMENT_ED)
                    else: # is MDD:
                        p.event(Event.TEST_9Q) # Note: This is in the flowchart diagram.
                        WARNING.print_once(k='SC_REPEATED_9Q_TEST', t="""
                            [!NOTE] SDDP 2013 will repeat the 9Q test twice in SC. It is in the flowchart diagram.
                            """)
                        if p._9q >= 7 and p._9q < 13: # Mild
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            Kongsuk.PC.followup_9Q(p)
                            
                        elif p._9q >= 13 and p._9q < 19: # Moderate
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            p.event(Event.TREATMENT_Rx)
                            SDDP2013.SC.followup_9Q_Rx(p)

                        elif p._9q >= 19: # Severe
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            p.event(Event.TREATMENT_Rx)
                            p.event(Event.REFER_TO_TC)
                            p.travel(Event.TRAVEL, CareLevel.TC)
                            SDDP2013.TC.receive_referral(p)
        @staticmethod
        def followup_9Q_Rx(p:Patient):
            p.event(Event.FOLLOWUP_SC_9Q_Rx)
            interval_months = 1
            duration_months = 6
            for i in range(1, duration_months+1, interval_months):
                p.travel(Event.TRAVEL, CareLevel.SC)
                p.event(Event.VISIT_SC)
                p.event(Event.TEST_9Q)
                p.event(Event.TREATMENT_Rx)


    class TC(DO):
        """ TC - (Tests: . Treatments: "According to Standards".  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_TC)
            SDDP2013.SC.followup_9Q_Rx(p)










class SDDP2013_NonDup9Q:
    """
    ### Surveillance System - SDDP 2013 Strict
    - thaidepression.com
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [SDDP2013] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            p.event(Event.NOTIFY_REPORT)
            if p._2q > 0 or p._9q > 0:
                p.event(Event.TREATMENT_ED)
                if p.is_SDDPReferToPC_vs_SC():
                    p.event(Event.REFER_TO_PC)
                    p.travel(Event.TRAVEL, CareLevel.PC)
                    SDDP2013_NonDup9Q.PC.receive_referral(p)
                else:
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    SDDP2013_NonDup9Q.SC.receive_referral(p)
            else:
                pass

    class PC(DO):
        """ PC - (Tests: 2Q, 9Q. Treatments: Ed, Report. Follow-up: 9Q. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_PC)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
            if is_viable_valid_9Q:
                if p._9q > 0 and p._9q < 7:
                    pass
                elif p._9q >= 7:
                    p.event(Event.TEST_8Q) # General Policy to exclude 8Q data results from analysis.
                    Kongsuk.PC.followup_9Q(p) # Note: This is in the flowchart diagram.
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    SDDP2013_NonDup9Q.SC.receive_referral(p)
                    

    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_SC)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                if p._9q >= 7:
                    p.event(Event.TEST_8Q)
                    p.event(Event.TEST_RO_PHYSICAL)
                    p.event(Event.TEST_RO_MEDICINE)
                    p.event(Event.TEST_MDD_DIAGNOSIS)
                    if not p.is_MDD():
                        p.event(Event.TEST_RO_PSYCHOSOCIAL)
                        if p.is_PsySocial():
                            p.event(Event.TREATMENT_CSG)
                            Kongsuk.PC.followup_9Q(p)
                        else:
                            p.event(Event.TREATMENT_ED)
                    else: # is MDD:
                        # p.event(Event.TEST_9Q) # Note: This is in the flowchart diagram.
                        # WARNING.print_once(k='SC_REPEATED_9Q_TEST', t="""
                        #     [!NOTE] SDDP 2013 will repeat the 9Q test twice in SC. It is in the flowchart diagram.
                        #     """)
                        if p._9q >= 7 and p._9q < 13: # Mild
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            Kongsuk.PC.followup_9Q(p)
                            
                        elif p._9q >= 13 and p._9q < 19: # Moderate
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            p.event(Event.TREATMENT_Rx)
                            SDDP2013.SC.followup_9Q_Rx(p)

                        elif p._9q >= 19: # Severe
                            p.event(Event.TREATMENT_CSG)
                            p.event(Event.TREATMENT_ED)
                            p.event(Event.TREATMENT_Rx)
                            p.event(Event.REFER_TO_TC)
                            p.travel(Event.TRAVEL, CareLevel.TC)
                            SDDP2013.TC.receive_referral(p)

class Kongsuk:
    """
    Kongsuk, et al., 2008.
    """
    class HV(DO):
        """ HV - (Tests: 2Q.  Treatments: . Follow-up: .) [Kongsuk etal, 2008] """
        @staticmethod
        def visits(p:Patient):
            p.travel(Event.TRAVEL_HV, CareLevel.PC)
            p.event(Event.VISIT_HV)
            p.event(Event.TEST_2Q)
            if p._2q > 0 or p._9q > 0:
                if p.is_SDDPReferToPC_vs_SC():
                    p.event(Event.REFER_TO_PC)
                    p.travel(Event.TRAVEL, CareLevel.PC)
                    Kongsuk.PC.receive_referral(p)
                else:
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    Kongsuk.SC.receive_referral(p)
            else:
                pass

    class PC(DO):
        """ PC - (Tests: 2Q, 9Q. Treatments: Ed, Report. Follow-up: 9Q. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_PC)
            p.event(Event.TREATMENT_ED)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)
            if is_viable_valid_9Q:
                if p._9q > 0 and p._9q < 7:
                    Kongsuk.PC.followup_9Q(p)
                elif p._9q >= 7:
                    p.event(Event.REFER_TO_SC)
                    p.travel(Event.TRAVEL, CareLevel.SC)
                    Kongsuk.SC.receive_referral(p)

        @staticmethod
        def followup_9Q(p:Patient):
            p.event(Event.FOLLOWUP_PC_9Q)
            interval_months = 1
            duration_months = 12
            duration_months = duration_months if not p.is_EarlyFollowUpDropout() else duration_months//2
            for i in range(1, duration_months+1, interval_months):
                p.travel(Event.TRAVEL, CareLevel.PC)
                p.event(Event.VISIT_PC)
                p.event(Event.TEST_9Q)

    class SC(DO):
        """ SC - (Tests: 2Q, 9Q, 8Q, Doctor Diagnosis. Treatments: Ed, CSG, Rx, Report.  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_SC)
            p.event(Event.TEST_9Q)
            is_viable_valid_9Q = not (p._9q <= 0 or p._9q > 27)

            if is_viable_valid_9Q:
                p.event(Event.TEST_8Q)
                p.event(Event.TEST_RO_PHYSICAL)
                p.event(Event.TEST_RO_MEDICINE)
                p.event(Event.TEST_MDD_DIAGNOSIS)
                if not p.is_MDD():
                    p.event(Event.TEST_RO_PSYCHOSOCIAL)
                    if p.is_PsySocial():
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        Kongsuk.PC.followup_9Q(p)
                    else:
                        p.event(Event.TREATMENT_ED)
                else: # is MDD:
                    if p._9q >= 7 and p._9q < 13: # Mild
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        Kongsuk.PC.followup_9Q(p)
                    elif p._9q >= 13 and p._9q < 19: # Moderate
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        p.event(Event.REFER_TO_TC)
                        p.travel(Event.TRAVEL, CareLevel.TC)
                        Kongsuk.TC.receive_referral(p)

                    elif p._9q >= 19: # Severe
                        p.event(Event.TREATMENT_CSG)
                        p.event(Event.TREATMENT_ED)
                        p.event(Event.TREATMENT_Rx)
                        p.event(Event.REFER_TO_TC)
                        p.travel(Event.TRAVEL, CareLevel.TC)
                        Kongsuk.TC.receive_referral(p)
        @staticmethod
        def followup_9Q_Rx(p:Patient):
            p.event(Event.FOLLOWUP_SC_9Q_Rx)
            interval_months = 1
            duration_months = 6
            for i in range(1, duration_months+1, interval_months):
                p.travel(Event.TRAVEL, CareLevel.SC)
                p.event(Event.VISIT_SC)
                p.event(Event.TEST_9Q)
                p.event(Event.TREATMENT_Rx)


    class TC(DO):
        """ TC - (Tests: . Treatments: "According to Standards".  Follow-up: 9Q-Rx. ) """
        @staticmethod
        def receive_referral(p:Patient):
            p.event(Event.VISIT_TC)
            Kongsuk.SC.followup_9Q_Rx(p)
