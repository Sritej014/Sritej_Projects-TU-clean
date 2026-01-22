/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2012-2016 OpenFOAM Foundation
    Copyright (C) 2018-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "volFields.H"

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //
using namespace Foam;
using namespace Foam::functionObjects;
template<class Type>
void Foam::functionObjects::turbulenceTotal::processField
(
    const word& fieldName,
    const tmp<GeometricField<Type, fvPatchField, volMesh>>& tvalue
)
{
    typedef GeometricField<Type, fvPatchField, volMesh> FieldType;

    // Build the fully scoped field name, e.g. "total_k"
    const word localName(IOobject::scopedName(prefix_, fieldName));

    // Attempt to find the field in the object registry
    FieldType* fldPtr = obr_.getObjectPtr<FieldType>(localName);

    if (fldPtr)
    {
        // If found, update its contents from tvalue
        (*fldPtr) == tvalue();
    }
    else
    {
        // Otherwise, create a new field and register it
        fldPtr = new FieldType
        (
            IOobject
            (
                localName,
                obr_.time().timeName(),
                obr_,
                IOobject::LAZY_READ,
                IOobject::NO_WRITE,
                IOobject::REGISTER
            ),
            tvalue
        );
        obr_.store(fldPtr);
    }
}




void turbulenceTotal::computeKResolved(volScalarField& kResolved) const
{
    // Lookup UPrime2Mean (a volSymmTensorField)
    const volSymmTensorField& uprime2mean =
        mesh_.lookupObject<volSymmTensorField>("UPrime2Mean");


    
    // Here we use .component() to extract diagonal entries.
    // In many OpenFOAM implementations, these may be accessed as .component(symmTensor::XX),
    // .component(symmTensor::YY) and .component(symmTensor::ZZ). For brevity, here we use indices:
    kResolved = 0.5 * (
        uprime2mean.component(0) +
        uprime2mean.component(1) +
        uprime2mean.component(2)
    );
}

void turbulenceTotal::computeTotalK
(
    const volScalarField& kResolved,
    const volScalarField& kSGS,
    volScalarField& totalK
) const
{
  
    totalK = kResolved + kSGS;
}



// ************************************************************************* //
