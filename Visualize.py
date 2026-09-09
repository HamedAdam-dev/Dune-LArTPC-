#############################
# Python imports
#############################
# Standard python imports
import random
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Creating a responsive 3D plot
import matplotlib
import shutil
import itertools
from mpl_toolkits.mplot3d import Axes3D
from termcolor import colored, cprint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

track_pdgs = [13, 2212, 211, 22, 11]
shower_pdgs = [11, 22]
pdgs = [13, 11, 2212, 211, 22]
pdgColours_graph = {13 : 'blue', 11 : 'red', 2212 : 'green', 211 : 'pink', 22 : 'tab:orange', 321 : 'tab:brown'}
pdgStrings_graph = {13 : 'muon', 11 : 'electron', 2212 : 'proton', 211 : 'pion', 22 : 'photon', 321 : 'kaon'}

trackShowerColours_graph = {11 : 'red', 13 : 'blue'}
trackShowerStrings_graph = {11 : 'shower', 13 : 'track'}



fileName = 'ccnutree_v3.root'

treeFile = uproot.open(fileName)
tree = treeFile['ccnuselection/ccnusel']
branches = tree.arrays()

# Event level info
event_main = np.array(branches['Event_Event'])
run_main = np.array(branches['Event_Run'])
subrun_main = np.array(branches['Event_Subrun'])
nuPDG_main = branches['Nu_True_Osc_PDG']
isNC_main = branches['Nu_True_IsNC']
nuTrueEnergy_main = branches['Nu_True_Energy']

# Event level visualisation
trueNuVertexX_main = branches['Nu_True_VertexX']
trueNuVertexY_main = branches['Nu_True_VertexY']
trueNuVertexZ_main = branches['Nu_True_VertexZ']
recoNuVertexX_main = branches['Nu_Reco_VertexX']
recoNuVertexY_main = branches['Nu_Reco_VertexY']
recoNuVertexZ_main = branches['Nu_Reco_VertexZ']

# PFP level info 
truePDG_main = branches['MatchedMC_PDG']
completeness_main = branches['PFP_Completeness']
purity_main = branches['PFP_Purity']
trackShowerScore_main = branches['PFP_TrackShowerScore']

# PFP level visualisation
recoVertexX_main = branches['PFP_VertexX']
recoVertexY_main = branches['PFP_VertexY']
recoVertexZ_main = branches['PFP_VertexZ']
spacePointsX_main = branches['PFP_SpacepointX']
spacePointsY_main = branches['PFP_SpacepointY']
spacePointsZ_main = branches['PFP_SpacepointZ']
recoID_main = branches['PFP_RecoID']

# #################################################################

# #################################################################

target_run = 1102
target_subrun = 1
target_event = 56001

'''
where(condition, [x, y], /)
Return elements chosen from `x` or `y` depending on `condition`.
'''

target_index = np.where(np.logical_and(event_main == target_event, np.logical_and(run_main == target_run, subrun_main == target_subrun)))

if (target_index[0].shape[0] == 0) :
    raise('Event not found :(')
    
print(target_index)
iEvent = target_index[0][0]
print('iEvent:', iEvent)

# ATTN USER: You need to set the event index number!
iEvent = 0
print('')
print('Run:', run_main[iEvent])
print('Subrun:', subrun_main[iEvent])
print('Event:', event_main[iEvent])
print(tree.keys())

######################################################
# Get event arrays 
######################################################
# Event level info
event_event = event_main[iEvent]
run_event = run_main[iEvent]
subrun_event = subrun_main[iEvent]
nuPDG_event = nuPDG_main[iEvent]
isNC_event = isNC_main[iEvent]
nuTrueEnergy_event = nuTrueEnergy_main[iEvent]

# Event level visualisation
trueNuVertexX_event = trueNuVertexX_main[iEvent]
trueNuVertexY_event = trueNuVertexY_main[iEvent]
trueNuVertexZ_event = trueNuVertexZ_main[iEvent]
recoNuVertexX_event = recoNuVertexX_main[iEvent]
recoNuVertexY_event = recoNuVertexY_main[iEvent]
recoNuVertexZ_event = recoNuVertexZ_main[iEvent]

# PFP level info 
truePDG_event = np.array(truePDG_main[iEvent])
completeness_event = np.array(completeness_main[iEvent])
purity_event = np.array(purity_main[iEvent])
trackShowerScore_event = np.array(trackShowerScore_main[iEvent])
pandoraPFPCode_event = np.zeros(trackShowerScore_event.shape, dtype='int')
pandoraPFPCode_event[trackShowerScore_event > 0.5] = 13
pandoraPFPCode_event[np.logical_and(np.logical_not(trackShowerScore_event > 0.5), trackShowerScore_event > 0.0)] = 11

# PFP level visualisation
recoVertexX_event = np.array(recoVertexX_main[iEvent])
recoVertexY_event = np.array(recoVertexY_main[iEvent])
recoVertexZ_event = np.array(recoVertexZ_main[iEvent])
spacePointsX_event = spacePointsX_main[iEvent]
spacePointsY_event = spacePointsY_main[iEvent]
spacePointsZ_event = spacePointsZ_main[iEvent]
recoID_event = recoID_main[iEvent]

# nParticles
nParticles = truePDG_event.shape[0]
#####################################################
# Set PFP colour -
#####################################################


colourMode = 1

############################################
# Set PFP colours - USER DOESN'T NEED TO ADD ANYTHING
############################################
particleColours = []
particleStrings = []

for iParticle in range(nParticles) : 
    
    truePDG_particle = abs(truePDG_event[iParticle])
    pandoraPFPCode_particle = pandoraPFPCode_event[iParticle]

    if (colourMode == 0) :
        particleColours.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
        particleStrings.append(str(recoID_event[iParticle]))
    elif (colourMode == 1) :
        particleColours.append(pdgColours_graph.get(abs(truePDG_particle), 'tab:gray'))
        particleStrings.append(pdgStrings_graph.get(abs(truePDG_particle), 'unknown'))
    elif (colourMode == 2) :
        particleColours.append(trackShowerColours_graph.get(abs(pandoraPFPCode_particle), 'tab:gray'))
        particleStrings.append(trackShowerStrings_graph.get(abs(pandoraPFPCode_particle), 'unknown'))
    else :
        raise('you picked an invalid option!')
    
###############################################################################################
# 3D RECONSTRUCTION 
###############################################################################################

fig = plt.figure(figsize=(9,9))
ax = plt.axes(projection='3d')

plt.title('(R:S:E): (' + str(run_event) + ", " + str(subrun_event) + ", " + str(event_event) + ")")
plt.xlabel('X')
plt.ylabel('Y')

for iParticle in range(nParticles) : 
    
    recoVertexX_particle = recoVertexX_event[iParticle]
    recoVertexY_particle = recoVertexY_event[iParticle]
    recoVertexZ_particle = recoVertexZ_event[iParticle]
    spacePointsX_particle = np.array(spacePointsX_event[iParticle])
    spacePointsY_particle = np.array(spacePointsY_event[iParticle])
    spacePointsZ_particle = np.array(spacePointsZ_event[iParticle])
    
    if (recoVertexZ_particle < -990) :
        continue
    
    handles, labels = ax.get_legend_handles_labels()
    
    if particleStrings[iParticle] in str(labels) :
        ax.scatter(spacePointsX_particle, spacePointsY_particle, spacePointsZ_particle, color = particleColours[iParticle], s = 5.0, alpha = 0.5)
    else :
        ax.scatter(spacePointsX_particle, spacePointsY_particle, spacePointsZ_particle, color = particleColours[iParticle], s = 5.0, alpha = 0.5, label=particleStrings[iParticle])
        
    ax.scatter(recoVertexX_particle, recoVertexY_particle, recoVertexZ_particle, color = particleColours[iParticle], s = 50.0, marker='*')    
    
#ax.scatter(recoNuVertexX_event, recoNuVertexY_event, recoNuVertexZ_event, color = "blue", s = 100.0, marker='*')
ax.scatter(trueNuVertexX_event, trueNuVertexY_event, trueNuVertexZ_event, color = "black", s = 100.0, marker='*')

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

print('-------------------------------------')
print(('NC' if isNC_event else 'CC'), ('numu' if (nuPDG_event == 14) else ('nue' if (nuPDG_event == 12) else 'nutau')), 'event')
print('True Neutrino Energy:', round(nuTrueEnergy_event, 2), "GeV")
print('True Neutrino Vertex:', '(' + str(round(trueNuVertexX_event, 2)) + ', ' + str(round(trueNuVertexY_event, 2)) + ', ' + str(round(trueNuVertexZ_event, 2)) + ') cm')
print('Reco Neutrino Vertex:', '(' + str(round(recoNuVertexX_event, 2)) + ', ' + str(round(recoNuVertexY_event, 2)) + ', ' + str(round(recoNuVertexZ_event, 2)) + ') cm')
print('-------------------------------------')

plt.legend()

plt.show()



fig, ax = plt.subplots()

for pdg in track_pdgs:
    values = np.array(ak.flatten(branches['Track_Length'][branches['MatchedMC_PDG'] == pdg])) # awkward arrays are such a pain
    weights = np.ones_like(values) * (1. / values.shape[0])
    ax.hist(values, bins=100, range=[0,1000], histtype='step', label=pdgStrings_graph[pdg], color=pdgColours_graph[pdg], weights=weights)
    
ax.set_title('Track Length')
ax.set_ylabel('Fraction of Particles')
ax.set_xlabel('Track Length [cm]')
ax.legend()

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import itertools


my_hash_map_2D = {
    'PFP_RecoID': [0, 15],
    'PFP_CurrentReco_IsPrimary': [0, 1],
    'PFP_CurrentReco_Generation': [0, 5],
    'PFP_CurrentReco_ParentRecoID': [0, 5000],
    'PFP_TrackShowerScore': [0, 1],
    'PFP_VertexX': [0, 450],
    'PFP_VertexY': [0, 1100],
    'PFP_VertexZ': [0, 1500],
    'PFP_N2DHits': [0, 5000],
    'PFP_NSpacepoints': [0, 5000],
    'PFP_Completeness': [0, 1.25],
    'PFP_Purity': [0, 2],
    'PFP_ShowerFitSuccess': [0, 2],
    'PFP_TrackFitSuccess': [0, 2],
    'Shower_StartX': [0, 400],
    'Shower_StartY': [0, 1200],
    'Shower_StartZ': [0, 1500],
    'Shower_Length': [0, 2000],
    'Track_DeflecAngleSD': [0, 100],
    'Track_Length': [0, 5000],
    'Track_EvalRatio': [0, 1],
    'Track_Concentration': [0, 110],
    'Track_CoreHaloRatio': [0, 120],
    'Track_Conicalness': [0, 80],
    'Track_dEdxStart': [0, 50],
    'Track_dEdxEnd': [0, 50],
    'Track_dEdxEndRatio': [0, 25], 
    'Shower_BestPlane': [0, 10],
}

my_hash_map = {
    'Shower_Energy': [0, 100],
    'Shower_InitialdEdx': [0, 20],
}

my_hash_map_1D = {
    'Nu_True_Unosc_PDG': [0, 20],
    'Nu_True_Osc_PDG': [0, 20],
    'Track_NTrajPoints': [0, 200],
    'Nu_True_SimID': [0, 1],
    'Nu_True_IsNC': [0, 10],
    'Nu_True_Mode': [0, 20],
    'Nu_True_TargetZ': [0, 40],
    'Nu_True_Energy': [0, 100],
    'Nu_True_MomX': [0, 1],
    'Nu_True_MomY': [0, 20],
    'Nu_True_MomZ': [0, 0],
    'Nu_True_VertexX': [0, 400],
    'Nu_True_VertexY': [0, 650],
    'Nu_True_VertexZ': [0, 1500],
    'Lep_True_PDG': [0, 10],
    'Lep_True_Energy': [0, 10],
    'Lep_True_MomX': [0, 4],
    'Lep_True_MomY': [0, 10],
    'Lep_True_MomZ': [0, 50],
    'Lep_True_EndX': [0, 400],
    'Lep_True_EndY': [0, 1],
    'Lep_True_EndZ': [0, 1500],
    'Lep_True_NuAngle': [0, 4],
    'Event_NMCParticles': [0, 1200],
    'Event_NRecoPFPs': [0, 50],
    'Nu_Reco_VertexX': [0, 400],
    'Nu_Reco_VertexY': [0, 800],
    'Nu_Reco_VertexZ': [0, 1500],
    'Nu_OscProb': [0, 2],
    'Event_Run': [0, 1500],
    'Event_Subrun': [0, 1.25],
    'Event_Event': [0, 5000],
}

def compute_dca(event_index, pfp_index,branches):
    x = np.asarray(branches['PFP_SpacepointX'][event_index][pfp_index])
    y = np.asarray(branches['PFP_SpacepointY'][event_index][pfp_index])
    z = np.asarray(branches['PFP_SpacepointZ'][event_index][pfp_index])

    if len(x) < 2 or not (len(x) == len(y) == len(z)):
        return None, None

    coords = np.column_stack((x, y, z))
    centered = coords - coords.mean(axis=0)
    direction = np.linalg.svd(centered, full_matrices=False)[2][0]

    vertex = np.array([
        branches['Nu_Reco_VertexX'][event_index],
        branches['Nu_Reco_VertexY'][event_index],
        branches['Nu_Reco_VertexZ'][event_index],
    ])
    shower_start = np.array([
        branches['Shower_StartX'][event_index][pfp_index],
        branches['Shower_StartY'][event_index][pfp_index],
        branches['Shower_StartZ'][event_index][pfp_index],
    ])

    w = shower_start - vertex
    r = w - np.dot(w, direction) * direction
    
    my_d=np.linalg.norm(r)
    my_displacment= np.linalg.norm(w)
    print( my_displacment )
    return my_d,my_displacment


def chisquare(h1, h2):
    denominator = h1 + h2
    mask = denominator > 0
    return 0.5 * np.sum((h1[mask] - h2[mask]) ** 2 / denominator[mask])
seperations = {}


for var_name, axis_range in my_hash_map_2D.items():
    fig, ax = plt.subplots()
    hists = {}
    for pdg in track_pdgs:
        values = np.array(ak.flatten(branches[var_name][branches['MatchedMC_PDG'] == pdg]))
        weights = np.ones_like(values, dtype=float) * (1. / values.shape[0])
        my_counts, _, _ = ax.hist(values, bins=100, range=axis_range, histtype='step',
                                  label=pdgStrings_graph[pdg], color=pdgColours_graph[pdg],
                                  weights=weights)
        hists[pdg] = my_counts
    seperations[var_name] = sum(
        chisquare(hists[a], hists[b]) for a, b in itertools.combinations(hists, 2)
    )
    ax.set_ylabel('Fraction of Particles')
    ax.set_xlabel(var_name)
    ax.legend()
   
for var_name, axis_range in my_hash_map.items():
    fig, ax = plt.subplots()
    hists = {}
    for pdg in track_pdgs:
        values = np.array(ak.flatten(branches[var_name][:, :, 2][branches['MatchedMC_PDG'] == pdg]))
        weights = np.ones_like(values, dtype=float) * (1. / values.shape[0])
        my_counts, _, _ = ax.hist(values, bins=100, range=axis_range, histtype='step',
                                  label=pdgStrings_graph[pdg], color=pdgColours_graph[pdg],
                                  weights=weights)
        hists[pdg] = my_counts
    seperations[var_name] = sum(
        chisquare(hists[a], hists[b]) for a, b in itertools.combinations(hists, 2)
    )
    ax.set_ylabel('Fraction of Particles')
    ax.set_xlabel(var_name)
    ax.legend()
    fig.savefig(f"{var_name}.png", dpi=300, bbox_inches="tight")
    

for var_name, axis_range in my_hash_map_1D.items():
    fig, ax = plt.subplots()
    
    for pdg in track_pdgs:
        values = np.array(branches[var_name]) # awkward arrays are such a pain
        weights = np.ones_like(values) * (1. / len(values))
        ax.hist(values, bins=100, range=axis_range , histtype='step', color=pdgColours_graph[pdg], weights=weights)
    ax.set_ylabel('Fraction of Particles')
    ax.set_xlabel(var_name)

ranked = dict(sorted(seperations.items(), key=lambda kv: kv[1], reverse=True))
for var_name, sep in ranked.items():
    print(f"{sep:.3f}  {var_name}")
#also save the truepdg
my_branches=[ 'PFP_TrackShowerScore', 'Track_EvalRatio', 'Shower_Length', 'Track_dEdxEnd',
    'Track_Length', 'PFP_NSpacepoints', 'Track_DeflecAngleSD', 'Track_Concentration',
    'PFP_N2DHits', 'Shower_InitialdEdx', 'Track_Conicalness', 'Track_CoreHaloRatio',
    'Track_dEdxStart', 'Shower_Energy', 'Track_dEdxEndRatio',
    'PFP_CurrentReco_Generation', 'dca']

# Plotting: bucket by PDG, reusing the same function
dca = {pdg: [] for pdg in track_pdgs}
for event_index in range(len(branches['PFP_SpacepointX'])):
    for pfp_index in range(len(branches['PFP_SpacepointX'][event_index])):
        my_pdg = branches['MatchedMC_PDG'][event_index][pfp_index]
        if my_pdg not in dca:
            continue                      
        d = compute_dca(event_index, pfp_index)
        if d is not None:
            dca[my_pdg].append(d)
print(dca)
ax.set_ylabel('Fraction of Particles')
ax.set_xlabel('dca')
ax.legend()


# BDT: aligned per-PFP jagged array 
dca_per_pfp = []
displacment_per_pfp=[]
for event_index in range(len(branches['PFP_SpacepointX'])):
    row_dca = []
    row_displacement = []
    
    for pfp_index in range(len(branches['PFP_SpacepointX'][event_index])):

        d,displacment=compute_dca(event_index, pfp_index,branches)#
        row_dca.append(d if d is not None else -1)
        row_displacement.append(displacment if displacment is not None else -1)
        
    dca_per_pfp.append(row_dca)
    displacment_per_pfp.append(row_displacement)
    
dca_ak = ak.Array(dca_per_pfp)
displacment_ak = ak.Array(displacment_per_pfp)

# Map each feature name to its per-PFP array, DCA included.
feature_arrays = {
    'PFP_TrackShowerScore':      branches['PFP_TrackShowerScore'],
    'Track_EvalRatio':           branches['Track_EvalRatio'],
    'Shower_Length':             branches['Shower_Length'],
    'Track_dEdxEnd':             branches['Track_dEdxEnd'],
    'Track_Length':              branches['Track_Length'],
    'PFP_NSpacepoints':          branches['PFP_NSpacepoints'],
    'Track_DeflecAngleSD':       branches['Track_DeflecAngleSD'],
    'Track_Concentration':       branches['Track_Concentration'],
    'PFP_N2DHits':               branches['PFP_N2DHits'],
    'Shower_InitialdEdx':        branches['Shower_InitialdEdx'][:, :, 2],  
    'Track_Conicalness':         branches['Track_Conicalness'],
    'Track_CoreHaloRatio':       branches['Track_CoreHaloRatio'],
    'Track_dEdxStart':           branches['Track_dEdxStart'],
    'Shower_Energy':             branches['Shower_Energy'][:, :, 2],       
    'Track_dEdxEndRatio':        branches['Track_dEdxEndRatio'],
    'PFP_CurrentReco_Generation':branches['PFP_CurrentReco_Generation'],
    'dca':                       dca_ak,
    'displament':                displacment_ak,
    'MatchedMC_PDG':             branches['MatchedMC_PDG'],  
}

var = [
    'PFP_TrackShowerScore', 'Track_EvalRatio', 'Shower_Length', 
    'Track_dEdxEnd', 'Track_Length', 'PFP_NSpacepoints', 
    'Track_DeflecAngleSD', 'Track_Concentration', 'PFP_N2DHits', 
    'Shower_InitialdEdx', 'Track_Conicalness', 'Track_CoreHaloRatio', 
    'Track_dEdxStart', 'Shower_Energy', 'Track_dEdxEndRatio', 
    'PFP_CurrentReco_Generation', 'dca', 'displament'
]

savez_dict = {}
for name, arr in feature_arrays.items():
    savez_dict[name] = np.asarray(ak.flatten(arr))

np.savez("t.npz", **savez_dict)   # once, outside the loop

# load back with keys that exist
data = np.load("t.npz")
