#ifndef TBFSMSTARPUUTILSMPI_HPP
#define TBFSMSTARPUUTILSMPI_HPP

#include <array>
#include <functional>
#include <map>
#include <mpi.h>
#include <starpu.h>
#include <starpu_mpi.h>
#include <vector>

class TbStarPUUtilsMPI {
protected:
  static void ExecOnWorkersBind(void *ptr) {
    std::function<void(void)> *func =
        static_cast<std::function<void(void)> *>(ptr);
    (*func)();
  }

public:
  static void ExecOnWorkers(const unsigned int inWorkersType,
                            std::function<void(void)> func) {
    starpu_execute_on_each_worker(ExecOnWorkersBind, &func, inWorkersType);
  }

  static void GlobalBarrier() {
    starpu_task_wait_for_all();
    starpu_mpi_barrier(MPI_COMM_WORLD);
  }

  static void EnsureDataConsistency(starpu_data_handle_t handle) {
    starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
  }
};

class TbfStarPUHandleBuilderMPI {
public:
  using CellHandleContainer =
      std::vector<std::vector<std::array<starpu_data_handle_t, 3>>>;
  using ParticleHandleContainer =
      std::vector<std::array<starpu_data_handle_t, 2>>;

  // Structure to manage distributed data ownership
  struct MPIDataInfo {
    int rank;
    int tag;
    bool isLocal;
  };

  using DataRegistry = std::map<starpu_data_handle_t, MPIDataInfo>;

  // Tag calculation for distributed data
  static int calculateTag(int level, int group, int dataType) {
    // Ensure unique tags for each data element
    const int MAX_GROUPS = 1000; // Adjust based on your application
    const int MAX_TYPES = 10;
    return level * MAX_GROUPS * MAX_TYPES + group * MAX_TYPES + dataType;
  }

  template <class TreeClass, class ConfigClass>
  static auto GetCellHandles(TreeClass &inTree, ConfigClass &inConfiguration,
                             int mpiRank, int mpiSize) {
    CellHandleContainer allCellHandles(inConfiguration.getTreeHeight());

    for (long int idxLevel = 0; idxLevel < inConfiguration.getTreeHeight();
         ++idxLevel) {
      auto &cellGroups = inTree.getCellGroupsAtLevel(idxLevel);
      auto currentCellGroup = cellGroups.begin();
      const auto endCellGroup = cellGroups.end();

      int groupIdx = 0;
      while (currentCellGroup != endCellGroup) {
        // Simple distribution: round-robin assignment of groups to ranks
        bool isLocal = (groupIdx % mpiSize == mpiRank);

        starpu_data_handle_t handleData;
        starpu_data_handle_t handleMultipole;
        starpu_data_handle_t handleLocal;

        if (isLocal) {
          // Local data - register normally
          starpu_variable_data_register(
              &handleData, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getDataPtr()),
              uint32_t(currentCellGroup->getDataSize()));

          starpu_variable_data_register(
              &handleMultipole, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getMultipolePtr()),
              uint32_t(currentCellGroup->getMultipoleSize()));

          starpu_variable_data_register(
              &handleLocal, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getLocalPtr()),
              uint32_t(currentCellGroup->getLocalSize()));
        } else {
          // Remote data - create placeholder handles
          starpu_variable_data_register(&handleData, -1, 0, 0);
          starpu_variable_data_register(&handleMultipole, -1, 0, 0);
          starpu_variable_data_register(&handleLocal, -1, 0, 0);
        }

        // Register with MPI
        int ownerRank = groupIdx % mpiSize;
        int tagData = calculateTag(idxLevel, groupIdx, 0);
        int tagMultipole = calculateTag(idxLevel, groupIdx, 1);
        int tagLocal = calculateTag(idxLevel, groupIdx, 2);

        starpu_mpi_data_register(handleData, tagData, ownerRank);
        starpu_mpi_data_register(handleMultipole, tagMultipole, ownerRank);
        starpu_mpi_data_register(handleLocal, tagLocal, ownerRank);

        std::array<starpu_data_handle_t, 3> cellHandles{
            handleData, handleMultipole, handleLocal};
        allCellHandles[idxLevel].push_back(cellHandles);

        ++currentCellGroup;
        ++groupIdx;
      }
    }
    return allCellHandles;
  }

  template <class TreeClass>
  static auto GetParticleHandles(TreeClass &inTree, int mpiRank, int mpiSize) {
    ParticleHandleContainer allParticlesHandles;

    auto &particleGroups = inTree.getParticleGroups();
    auto currentParticleGroup = particleGroups.begin();
    const auto endParticleGroup = particleGroups.end();

    int groupIdx = 0;
    while (currentParticleGroup != endParticleGroup) {
      // Simple distribution: round-robin assignment of groups to ranks
      bool isLocal = (groupIdx % mpiSize == mpiRank);

      starpu_data_handle_t handleData;
      starpu_data_handle_t handleRhs;

      if (isLocal) {
        // Local particles - register normally
        starpu_variable_data_register(
            &handleData, STARPU_MAIN_RAM,
            uintptr_t(currentParticleGroup->getDataPtr()),
            uint32_t(currentParticleGroup->getDataSize()));

        starpu_variable_data_register(
            &handleRhs, STARPU_MAIN_RAM,
            uintptr_t(currentParticleGroup->getRhsPtr()),
            uint32_t(currentParticleGroup->getRhsSize()));
      } else {
        // Remote particles - create placeholder handles
        starpu_variable_data_register(&handleData, -1, 0, 0);
        starpu_variable_data_register(&handleRhs, -1, 0, 0);
      }

      // Register with MPI
      int ownerRank = groupIdx % mpiSize;
      int tagData = calculateTag(inTree.getTreeHeight(), groupIdx, 0);
      int tagRhs = calculateTag(inTree.getTreeHeight(), groupIdx, 1);

      starpu_mpi_data_register(handleData, tagData, ownerRank);
      starpu_mpi_data_register(handleRhs, tagRhs, ownerRank);

      std::array<starpu_data_handle_t, 2> particlesHandles{handleData,
                                                           handleRhs};
      allParticlesHandles.push_back(particlesHandles);

      ++currentParticleGroup;
      ++groupIdx;
    }
    return allParticlesHandles;
  }

  static void CleanCellHandles(CellHandleContainer &inCellHandles) {
    for (auto &handlePerLevel : inCellHandles) {
      for (auto &handleGroup : handlePerLevel) {
        for (auto &handle : handleGroup) {
          if (handle != nullptr) {
            starpu_mpi_data_unregister(handle);
            starpu_data_unregister(handle);
          }
        }
      }
    }
  }

  static void CleanParticleHandles(ParticleHandleContainer &inParticleHandles) {
    for (auto &handleGroup : inParticleHandles) {
      for (auto &handle : handleGroup) {
        if (handle != nullptr) {
          starpu_mpi_data_unregister(handle);
          starpu_data_unregister(handle);
        }
      }
    }
  }
};

class TbfStarPUHandleBuilderTsmMPI {
public:
  using CellSrcHandleContainer =
      std::vector<std::vector<std::array<starpu_data_handle_t, 2>>>;
  using ParticleSrcHandleContainer =
      std::vector<std::array<starpu_data_handle_t, 1>>;
  using CellTgtHandleContainer =
      std::vector<std::vector<std::array<starpu_data_handle_t, 2>>>;
  using ParticleTgtHandleContainer =
      std::vector<std::array<starpu_data_handle_t, 2>>;

  static int calculateTag(int level, int group, int dataType) {
    const int MAX_GROUPS = 1000;
    const int MAX_TYPES = 10;
    return level * MAX_GROUPS * MAX_TYPES + group * MAX_TYPES + dataType;
  }

  template <class TreeClass, class ConfigClass>
  static auto GetCellSrcHandles(TreeClass &inTree, ConfigClass &inConfiguration,
                                int mpiRank, int mpiSize) {
    CellSrcHandleContainer allCellHandles(inConfiguration.getTreeHeight());

    for (long int idxLevel = 0; idxLevel < inConfiguration.getTreeHeight();
         ++idxLevel) {
      auto &cellGroups = inTree.getCellGroupsAtLevelSource(idxLevel);
      auto currentCellGroup = cellGroups.begin();
      const auto endCellGroup = cellGroups.end();

      int groupIdx = 0;
      while (currentCellGroup != endCellGroup) {
        bool isLocal = (groupIdx % mpiSize == mpiRank);

        starpu_data_handle_t handleData;
        starpu_data_handle_t handleMultipole;

        if (isLocal) {
          starpu_variable_data_register(
              &handleData, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getDataPtr()),
              uint32_t(currentCellGroup->getDataSize()));

          starpu_variable_data_register(
              &handleMultipole, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getMultipolePtr()),
              uint32_t(currentCellGroup->getMultipoleSize()));
        } else {
          starpu_variable_data_register(&handleData, -1, 0, 0);
          starpu_variable_data_register(&handleMultipole, -1, 0, 0);
        }

        int ownerRank = groupIdx % mpiSize;
        int tagData = calculateTag(idxLevel, groupIdx, 0);
        int tagMultipole = calculateTag(idxLevel, groupIdx, 1);

        starpu_mpi_data_register(handleData, tagData, ownerRank);
        starpu_mpi_data_register(handleMultipole, tagMultipole, ownerRank);

        std::array<starpu_data_handle_t, 2> cellHandles{handleData,
                                                        handleMultipole};
        allCellHandles[idxLevel].push_back(cellHandles);

        ++currentCellGroup;
        ++groupIdx;
      }
    }
    return allCellHandles;
  }

  template <class TreeClass, class ConfigClass>
  static auto GetCellTgtHandles(TreeClass &inTree, ConfigClass &inConfiguration,
                                int mpiRank, int mpiSize) {
    CellTgtHandleContainer allCellHandles(inConfiguration.getTreeHeight());

    for (long int idxLevel = 0; idxLevel < inConfiguration.getTreeHeight();
         ++idxLevel) {
      auto &cellGroups = inTree.getCellGroupsAtLevelTarget(idxLevel);
      auto currentCellGroup = cellGroups.begin();
      const auto endCellGroup = cellGroups.end();

      int groupIdx = 0;
      while (currentCellGroup != endCellGroup) {
        bool isLocal = (groupIdx % mpiSize == mpiRank);

        starpu_data_handle_t handleData;
        starpu_data_handle_t handleLocal;

        if (isLocal) {
          starpu_variable_data_register(
              &handleData, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getDataPtr()),
              uint32_t(currentCellGroup->getDataSize()));

          starpu_variable_data_register(
              &handleLocal, STARPU_MAIN_RAM,
              uintptr_t(currentCellGroup->getLocalPtr()),
              uint32_t(currentCellGroup->getLocalSize()));
        } else {
          starpu_variable_data_register(&handleData, -1, 0, 0);
          starpu_variable_data_register(&handleLocal, -1, 0, 0);
        }

        int ownerRank = groupIdx % mpiSize;
        int tagData = calculateTag(idxLevel, groupIdx, 0);
        int tagLocal = calculateTag(idxLevel, groupIdx, 2);

        starpu_mpi_data_register(handleData, tagData, ownerRank);
        starpu_mpi_data_register(handleLocal, tagLocal, ownerRank);

        std::array<starpu_data_handle_t, 2> cellHandles{handleData,
                                                        handleLocal};
        allCellHandles[idxLevel].push_back(cellHandles);

        ++currentCellGroup;
        ++groupIdx;
      }
    }
    return allCellHandles;
  }

  template <class TreeClass>
  static auto GetParticleSrcHandles(TreeClass &inTree, int mpiRank,
                                    int mpiSize) {
    ParticleSrcHandleContainer allParticlesHandles;

    auto &particleGroups = inTree.getParticleGroupsSource();
    auto currentParticleGroup = particleGroups.begin();
    const auto endParticleGroup = particleGroups.end();

    int groupIdx = 0;
    while (currentParticleGroup != endParticleGroup) {
      bool isLocal = (groupIdx % mpiSize == mpiRank);

      starpu_data_handle_t handleData;

      if (isLocal) {
        starpu_variable_data_register(
            &handleData, STARPU_MAIN_RAM,
            uintptr_t(currentParticleGroup->getDataPtr()),
            uint32_t(currentParticleGroup->getDataSize()));
      } else {
        starpu_variable_data_register(&handleData, -1, 0, 0);
      }

      int ownerRank = groupIdx % mpiSize;
      int tagData = calculateTag(inTree.getTreeHeight(), groupIdx, 0);

      starpu_mpi_data_register(handleData, tagData, ownerRank);

      std::array<starpu_data_handle_t, 1> particlesHandles{handleData};
      allParticlesHandles.push_back(particlesHandles);

      ++currentParticleGroup;
      ++groupIdx;
    }
    return allParticlesHandles;
  }

  template <class TreeClass>
  static auto GetParticleTgtHandles(TreeClass &inTree, int mpiRank,
                                    int mpiSize) {
    ParticleTgtHandleContainer allParticlesHandles;

    auto &particleGroups = inTree.getParticleGroupsTarget();
    auto currentParticleGroup = particleGroups.begin();
    const auto endParticleGroup = particleGroups.end();

    int groupIdx = 0;
    while (currentParticleGroup != endParticleGroup) {
      bool isLocal = (groupIdx % mpiSize == mpiRank);

      starpu_data_handle_t handleData;
      starpu_data_handle_t handleRhs;

      if (isLocal) {
        starpu_variable_data_register(
            &handleData, STARPU_MAIN_RAM,
            uintptr_t(currentParticleGroup->getDataPtr()),
            uint32_t(currentParticleGroup->getDataSize()));

        starpu_variable_data_register(
            &handleRhs, STARPU_MAIN_RAM,
            uintptr_t(currentParticleGroup->getRhsPtr()),
            uint32_t(currentParticleGroup->getRhsSize()));
      } else {
        starpu_variable_data_register(&handleData, -1, 0, 0);
        starpu_variable_data_register(&handleRhs, -1, 0, 0);
      }

      int ownerRank = groupIdx % mpiSize;
      int tagData = calculateTag(inTree.getTreeHeight(), groupIdx, 0);
      int tagRhs = calculateTag(inTree.getTreeHeight(), groupIdx, 1);

      starpu_mpi_data_register(handleData, tagData, ownerRank);
      starpu_mpi_data_register(handleRhs, tagRhs, ownerRank);

      std::array<starpu_data_handle_t, 2> particlesHandles{handleData,
                                                           handleRhs};
      allParticlesHandles.push_back(particlesHandles);

      ++currentParticleGroup;
      ++groupIdx;
    }
    return allParticlesHandles;
  }

  template <class AnyCellHandleContainer>
  static void CleanCellHandles(AnyCellHandleContainer &inCellHandles) {
    for (auto &handlePerLevel : inCellHandles) {
      for (auto &handleGroup : handlePerLevel) {
        for (auto &handle : handleGroup) {
          if (handle != nullptr) {
            starpu_mpi_data_unregister(handle);
            starpu_data_unregister(handle);
          }
        }
      }
    }
  }

  template <class AnyParticleHandleContainer>
  static void
  CleanParticleHandles(AnyParticleHandleContainer &inParticleHandles) {
    for (auto &handleGroup : inParticleHandles) {
      for (auto &handle : handleGroup) {
        if (handle != nullptr) {
          starpu_mpi_data_unregister(handle);
          starpu_data_unregister(handle);
        }
      }
    }
  }
};

#endif
