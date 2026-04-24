
from collections import BitSet
from sys import size_of, num_logical_cores
from sys.ffi import external_call, get_errno


@fieldwise_init
struct Cpus(
    Movable,
    Copyable,
    Writable,
    Stringable,
    Sized
):
    var cpus: List[Cpu]

    fn __init__(out self):
        self.cpus = List[Cpu]()

    @staticmethod
    fn get_all(out self: Self) raises:

        self = Cpus()

        with open("/proc/cpuinfo", "r") as f:

            var processor: Optional[Int] = None
            var physical_id: Optional[Int] = None
            var core_id: Optional[Int] = None

            for line in f.read().split('\n'):

                # blank lines follow virtual core sections: build the virtual core
                if line == "":

                    # make sure we have all the numbers
                    if processor is None:
                        # double blank lines means EOF
                        break
                    if physical_id is None:
                        raise Error("missing physical_id")
                    if core_id is None:
                        raise Error("missing core_id")
                    
                    var virtual_core = VirtualCore(
                        cpu_id = physical_id.value(),
                        physical_id = core_id.value(),
                        virtual_id = processor.value()
                    )

                    # find or make the cpu
                    var cpu_index = self.index_of(cpu_id=virtual_core.cpu_id)
                    if cpu_index is None:
                        cpu_index = len(self.cpus)
                        self.cpus.append(Cpu(
                            cpu_id = virtual_core.cpu_id,
                            physical_cores = List[PhysicalCore]()
                        ))

                    ref cpu = self.cpus[cpu_index.value()]

                    # find or make the physical core
                    var pcore_index = cpu.index_of(physical_id=virtual_core.physical_id)
                    if pcore_index is None:
                        pcore_index = len(cpu.physical_cores)
                        cpu.physical_cores.append(PhysicalCore(
                            cpu_id = cpu.cpu_id,
                            physical_id = virtual_core.physical_id,
                            virtual_cores = List[VirtualCore]()
                        ))

                    ref pcore = cpu.physical_cores[pcore_index.value()]

                    pcore.virtual_cores.append(virtual_core^)
                    
                    # cleanup for the next section
                    processor = None
                    physical_id = None
                    core_id = None

                else:

                    # lines should look like `a : b` 
                    var parts = line.split(':')
                    for i in range(len(parts)):
                        parts[i] = parts[i].strip()
                    if len(parts) != 2:
                        continue

                    # read values from the virtual core
                    if parts[0] == 'processor':
                        processor = Int(parts[1])
                    elif parts[0] == 'physical id':
                        physical_id = Int(parts[1])
                    elif parts[0] == 'core id':
                        core_id = Int(parts[1])

    @staticmethod
    fn get_allowed(out cpus: Cpus) raises:
        cpus = Self.get_all()
        cpus.retain(virtual_core_ids = VirtualCore.read_allowed_ids())

    @staticmethod
    fn get_affinity(out cpus: Cpus) raises:
        cpus = Self.get_all()
        cpus.retain(virtual_core_ids = VirtualCoreSet.get_affinity().to_virtual_core_ids())

    fn __len__(self) -> Int:
        return len(self.cpus)

    fn __getitem__(self, i: Int) -> ref [self.cpus] Cpu:
        return self.cpus[i]

    fn index_of(self, *, cpu_id: Int) -> Optional[Int]:
        for i in range(len(self.cpus)):
            if self.cpus[i].cpu_id == cpu_id:
                return i
        return None
    
    fn retain(
        mut self,
        *,
        virtual_core_ids: List[Int]
    ):
        var disallowed_cpu_indices = List[Int]()
        for ci in range(len(self.cpus)):

            self.cpus[ci].retain(virtual_core_ids=virtual_core_ids)
            
            if len(self.cpus[ci].physical_cores) <= 0:
                disallowed_cpu_indices.append(ci)
        
        for ci in reversed(disallowed_cpu_indices):
            _ = self.cpus.pop(ci)

    fn physical_cores(
        self,
        out pcores: List[PhysicalCore]
    ):
        pcores = List[PhysicalCore]()
        for cpu in self.cpus:
            for pcore in cpu.physical_cores:
                pcores.append(pcore.copy())

    fn virtual_cores(
        self,
        out vcores: List[VirtualCore],
        *,
        even: Bool = True,
        odd: Bool = True
    ):
        vcores = List[VirtualCore]()
        for cpu in self.cpus:
            for pcore in cpu.physical_cores:
                for vcore in pcore.virtual_cores:
                    if (vcore.is_even() and even) or (not vcore.is_even() and odd):
                        vcores.append(vcore)

    fn virtual_core_ids(self, out vcores: List[Int]):
        vcores = List[Int]()
        for cpu in self.cpus:
            for pcore in cpu.physical_cores:
                for vcore in pcore.virtual_cores:
                    vcores.append(vcore.virtual_id)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('\n'.join(self.cpus))

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct Cpu(
    Movable,
    Copyable,
    Writable,
    Stringable
):
    var cpu_id: Int
    var physical_cores: List[PhysicalCore]

    fn retain(mut self, *, virtual_core_ids: List[Int]):
        
        var disallowed_pcore_indices = List[Int]()
        for pi in range(len(self.physical_cores)):

            var disallowed_vcore_indices = List[Int]()
            for vi in range(len(self.physical_cores[pi].virtual_cores)):
                if self.physical_cores[pi].virtual_cores[vi].virtual_id not in virtual_core_ids:
                    disallowed_vcore_indices.append(vi)

            for vi in reversed(disallowed_vcore_indices):
                _ = self.physical_cores[pi].virtual_cores.pop(vi)

            if len(self.physical_cores[pi].virtual_cores) <= 0:
                disallowed_pcore_indices.append(pi)
        
        for pi in reversed(disallowed_pcore_indices):
            _ = self.physical_cores.pop(pi)
        
    fn index_of(self, *, physical_id: Int) -> Optional[Int]:
        for i in range(len(self.physical_cores)):
            ref pcore = self.physical_cores[i]
            if pcore.physical_id == physical_id:
                return i
        return None

    fn virtual_cores(self, out vcores: List[VirtualCore]):
        vcores = List[VirtualCore]()
        for pcore in self.physical_cores:
            for vcore in pcore.virtual_cores:
                vcores.append(vcore)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('Cpu[',
            'id=', self.cpu_id,
            ', physical_cores=[\n  ', ",\n  ".join(self.physical_cores), '\n]',
        ']')

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct PhysicalCore(
    Movable,
    Copyable,
    Writable,
    Stringable
):
    var cpu_id: Int
    var physical_id: Int
    var virtual_cores: List[VirtualCore]

    fn index_of(self, *, virtual_id: Int) -> Optional[Int]:
        for i in range(len(self.virtual_cores)):
            ref vcore = self.virtual_cores[i]
            if vcore.virtual_id == virtual_id:
                return i
        return None

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('PhysicalCore[',
            'cpu=', self.cpu_id,
            ', id=', self.physical_id,
            ', virtual_cores=[', ', '.join(self.virtual_cores), ']'
        ']')

    fn __str__(self) -> String:
        return String.write(self)


@fieldwise_init
struct VirtualCore(
    Movable,
    ImplicitlyCopyable,
    Writable,
    Stringable,
    EqualityComparable
):
    var cpu_id: Int
    var physical_id: Int
    var virtual_id: Int

    @staticmethod
    fn read_allowed_ids(out virtual_ids: List[Int]) raises:
        """
        NOTE: sometimes more vcores are allowed than exist, due to mask sizes I guess.
        """

        virtual_ids = List[Int]()

        with open(String("/proc/", process_id(), "/status"), "r") as f:
            for line in f.read().split('\n'):
                comptime prefix = "Cpus_allowed_list:"
                if line.startswith(prefix):

                    # lines look like: `0-4,9` and `0-2,7,12-14`
                    parts = line[len(prefix):].strip().split(',')
                    for part in parts:
                        nums = part.split('-')
                        if len(nums) == 1:
                            virtual_ids.append(Int(nums[0]))
                        elif len(nums) == 2:
                            var start = Int(nums[0])
                            var stop = Int(nums[1])
                            for i in range(start, stop + 1):
                                virtual_ids.append(i)

    fn is_even(self) -> Bool:
        return self.virtual_id % 2 == 0

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('VirtualCore[',
            'cpu=', self.cpu_id,
            ', phy=', self.physical_id,
            ', vir=', self.virtual_id,
        ']')

    fn __str__(self) -> String:
        return String.write(self)

    fn __eq__(self, other: Self) -> Bool:
        return self.cpu_id == other.cpu_id
            and self.physical_id == other.physical_id
            and self.virtual_id == other.virtual_id


struct VirtualCoreSet(
    Movable,
    Copyable,
    Writable,
    Stringable
):
    var _mask: BitSet[Self._num_logical_cores]

    comptime _num_logical_cores = 1024
    # mask size must be known at compile time, and be at least as large as the one the kernel uses
    comptime _bitset_size = size_of[InlineArray[Int64, BitSet[Self._num_logical_cores]._words_size]]()

    fn __init__(out self):
        self._mask = BitSet[Self._num_logical_cores]()

    fn __init__(out self, vcores: List[VirtualCore]):
        self = self.__init__()
        for vcore in vcores:
            self.add(vcore)

    fn __init__(out self, vcore_ids: List[Int]):
        self = self.__init__()
        for id in vcore_ids:
            self.add(id)

    fn to_virtual_core_ids(self, out vcore_ids: List[Int]):
        vcore_ids = List[Int]()
        for i in range(Self._num_logical_cores):
            if self._mask.test(i):
                vcore_ids.append(i)

    fn __contains__(self, vcore: VirtualCore) -> Bool:
        return vcore.virtual_id in self

    fn __contains__(self, vcore_id: Int) -> Bool:
        return self._mask.test(vcore_id)

    fn add(mut self, vcore: VirtualCore):
        self.add(vcore.virtual_id)

    fn add(mut self, vcore_id: Int):
        self._mask.set(vcore_id)

    fn remove(mut self, vcore: VirtualCore):
        self.remove(vcore.virtual_id)

    fn remove(mut self, vcore_id: Int):
        self._mask.clear(vcore_id)

    # https://www.man7.org/linux/man-pages/man2/sched_setaffinity.2.html

    fn set_affinity(self) raises:
        var ret = external_call[
            "sched_setaffinity",
            Int32
        ](
            0,
            Self._bitset_size,
            self._mask._words.unsafe_ptr()
        )
        if ret == -1:
            raise Error("Failed to set thread affinity: ", get_errno())

    @staticmethod
    fn get_affinity(out self: Self) raises:
        self = Self()
        var ret = external_call[
            "sched_getaffinity",
            Int32
        ](
            0,
            Self._bitset_size,
            self._mask._words.unsafe_ptr()
        )
        if ret == -1:
            raise Error("Failed to get thread affinity: ", get_errno())

    fn write_to[W: Writer](self, mut writer: W):
        writer.write('VirtualCoreSet[', ', '.join(self.to_virtual_core_ids()), ']')

    fn __str__(self) -> String:
        return String.write(self)
